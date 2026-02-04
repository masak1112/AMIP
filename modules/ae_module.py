import lightning as L
import torch

from common.loss import latitude_weighted_rmse, WeightedLoss
from common.plotting import plot_reconstruction, plot_spectrum
from data.amip import SURFACE_VARIABLES, MULTILEVEL_VARIABLES, DIAGNOSTIC_VARIABLES

class AutoencoderModule(L.LightningModule):
    def __init__(self,
                 config: dict,
                 normalizer= None):
        '''
        TrainModule
        args:
            config (dict): configuration dictionary containing model, training and data configurations
            normalizer (object, optional): normalizer object for scaling input data. Defaults to None.
        '''

        super().__init__()
        self.config=config
        self.modelconfig = config['model']
        self.model_name = self.modelconfig["model_name"]
        self.lr = self.modelconfig["lr"]
        self.log_dir = config['training']['log_dir']
        self.downsample_levels = config['data'].get('downsample_levels', 1)

        self.criterion = WeightedLoss(latitude_resolution=180,
                                      longitude_resolution=360,
                                      nlevels = 26 // self.downsample_levels)
        self.n = normalizer

        self.history = False
        if self.model_name == "DCAE":
            from modules.models.AE import Encoder, Decoder
            self.encoder = Encoder(**self.modelconfig["DCAE"]["encoder"])
            self.decoder = Decoder(**self.modelconfig["DCAE"]["decoder"])
        elif self.model_name == "AE_3D":
            from modules.models.AE import Encoder3D, Decoder3D
            self.encoder = Encoder3D(**self.modelconfig["AE_3D"]["encoder"])
            self.decoder = Decoder3D(**self.modelconfig["AE_3D"]["decoder"])
        elif self.model_name == "AE_simple":
            from modules.models.AE_simple import Encoder, Decoder 
            self.encoder = Encoder(**self.modelconfig["AE_simple"]["encoder"])
            self.decoder = Decoder(**self.modelconfig["AE_simple"]["decoder"])
        elif self.model_name == "AE_simple_3D":
            from modules.models.AE_simple import Encoder3D, Decoder3D
            self.encoder = Encoder3D(**self.modelconfig["AE_simple_3D"]["encoder"])
            self.decoder = Decoder3D(**self.modelconfig["AE_simple_3D"]["decoder"])
        elif self.model_name == "AE_Decoder_Only":
            from modules.models.AE_simple import BilinearEncoder, Decoder 
            self.encoder = BilinearEncoder(**self.modelconfig["AE_Decoder_Only"]["encoder"])
            self.decoder = Decoder(**self.modelconfig["AE_Decoder_Only"]["decoder"])
        elif self.model_name == "AE_Atlas":
            from modules.models.AE_simple import BilinearEncoder
            from modules.models.AE_attn import NattenCombineDiT
            self.encoder = BilinearEncoder(**self.modelconfig["AE_Atlas"]["encoder"])
            self.decoder = NattenCombineDiT(**self.modelconfig["AE_Atlas"]["decoder"])
            self.history = True
        else:
            raise NotImplementedError(f"Model {self.model_name} not implemented")

        if config['training']['strategy'] == 'ddp' or config['training']['strategy'] == 'ddp_find_unused_parameters_true':
            self.ddp = True
        else:
            self.ddp = False

        self.save_hyperparameters()

    def forward(self, surface, multilevel, diagnostic):
        z_surface, z_multilevel, z_diagnostic = self.encoder(surface, multilevel, diagnostic)
        surface_pred, multilevel_pred, diagnostic_pred = self.decoder(z_surface, z_multilevel, z_diagnostic)

        return surface_pred, multilevel_pred, diagnostic_pred
    
    def forward_history(self, surface_history, multilevel_history, diagnostic_history,
                        surface, multilevel, diagnostic):
        
        z_surface, z_multilevel, z_diagnostic = self.encoder(surface, multilevel, diagnostic)

        surface_pred, multilevel_pred, diagnostic_pred = self.decoder(surface_history, multilevel_history, diagnostic_history,
                                                                     z_surface, z_multilevel, z_diagnostic)
        
        return surface_pred, multilevel_pred, diagnostic_pred
    
    def compute_loss(self, 
                     surface_pred, surface_target,
                     multilevel_pred, multilevel_target,
                     diagnostic_pred, diagnostic_target):
        
        return self.criterion(surface_pred, surface_target,
                              multilevel_pred, multilevel_target,
                              diagnostic_pred, diagnostic_target)
    
    def training_step(self, batch, batch_idx):
        
        if not self.history:
            surface_data = batch['surface'][:, 0] # b nlat nlon c
            multilevel_data = batch['multilevel'][:, 0] # b nlevel nlat nlon c
            diagnostic_data = batch['diagnostic'][:, 0] # b nlat nlon c

            surface_pred, multilevel_pred, diagnostic_pred = self.forward(surface_data, multilevel_data, diagnostic_data)
        else:
            surface_history = batch['surface'][:, 0] # b nlat nlon c
            multilevel_history = batch['multilevel'][:, 0]
            diagnostic_history = batch['diagnostic'][:, 0]

            surface_data = batch['surface'][:, 1] # b nlat nlon c
            multilevel_data = batch['multilevel'][:, 1]
            diagnostic_data = batch['diagnostic'][:, 1]

            surface_pred, multilevel_pred, diagnostic_pred = self.forward_history(
                surface_history, multilevel_history, diagnostic_history,
                surface_data, multilevel_data, diagnostic_data)

        loss = self.compute_loss(surface_pred, surface_data,
                                multilevel_pred, multilevel_data,
                                diagnostic_pred, diagnostic_data)

        self.log("train/loss", loss, on_step=True, on_epoch=True, sync_dist=self.ddp)

        return loss 

    def validation_step(self, batch, batch_idx): 
        
        if not self.history:
            surface_data = batch['surface'][:, 0] # b nlat nlon c
            multilevel_data = batch['multilevel'][:, 0] # b nlevel nlat nlon c
            diagnostic_data = batch['diagnostic'][:, 0] # b nlat nlon c

            surface_pred, multilevel_pred, diagnostic_pred = self.forward(surface_data, multilevel_data, diagnostic_data)
        else:
            surface_history = batch['surface'][:, 0] # b nlat nlon c
            multilevel_history = batch['multilevel'][:, 0]
            diagnostic_history = batch['diagnostic'][:, 0]

            surface_data = batch['surface'][:, 1] # b nlat nlon c
            multilevel_data = batch['multilevel'][:, 1]
            diagnostic_data = batch['diagnostic'][:, 1]

            surface_pred, multilevel_pred, diagnostic_pred = self.forward_history(
                surface_history, multilevel_history, diagnostic_history,
                surface_data, multilevel_data, diagnostic_data)

        loss_dict, pred_dict, data_dict = self.compute_loss_val(surface_pred, surface_data,
                                                multilevel_pred, multilevel_data,
                                                diagnostic_pred, diagnostic_data)
        
        self.log_losses(loss_dict)
        
        if batch_idx == 0: # only plot 1st batch
            if not self.ddp or self.global_rank == 0: # only run plotting on one gpu
                self.plot_predictions(pred_dict, data_dict)
    
    @torch.no_grad()
    def compute_loss_val(self,
                surface_pred, surface_data,
                multilevel_pred, multilevel_data,
                diagnostic_pred, diagnostic_data):
        
        surface_pred = self.n.denormalize_surface(surface_pred)
        multilevel_pred = self.n.denormalize_multilevel(multilevel_pred)
        diagnostic_pred = self.n.denormalize_diagnostic(diagnostic_pred)
        surface_data = self.n.denormalize_surface(surface_data)
        multilevel_data = self.n.denormalize_multilevel(multilevel_data)
        diagnostic_data = self.n.denormalize_diagnostic(diagnostic_data)

        pred_feat_dict = {}
        target_feat_dict = {}

        for c, surface_feat_name in enumerate(SURFACE_VARIABLES):
            pred_feat_dict[surface_feat_name] = surface_pred[..., c] # b nlat nlon
            target_feat_dict[surface_feat_name] = surface_data[..., c]

        for c, multilevel_feat_name in enumerate(MULTILEVEL_VARIABLES):
            pred_feat_dict[multilevel_feat_name] = multilevel_pred[..., c] # b nlevel nlat nlon 
            target_feat_dict[multilevel_feat_name] = multilevel_data[..., c]

        for c, diagnostic_feat_name in enumerate(DIAGNOSTIC_VARIABLES):
            pred_feat_dict[diagnostic_feat_name] = diagnostic_pred[..., c] # b nlat nlon
            target_feat_dict[diagnostic_feat_name] = diagnostic_data[..., c]

        nlat, nlon = surface_data.shape[1], surface_data.shape[2]
        loss_dict = {k:
                        latitude_weighted_rmse(pred_feat_dict[k], target_feat_dict[k],
                                                nlon=nlon, nlat=nlat, with_time=False
                                                ) for k in pred_feat_dict.keys()} # b or b l for each key
        
        return loss_dict, pred_feat_dict, target_feat_dict

    
    def plot_predictions(self, pred_feat_dict, target_feat_dict):

        t2m_pred = pred_feat_dict['2m_temperature'][0].cpu() #b h w -> h w 
        t2m_target = target_feat_dict['2m_temperature'][0].cpu()
        pr_6h_pred = pred_feat_dict['PRATEsfc'][0].cpu()
        pr_6h_target = target_feat_dict['PRATEsfc'][0].cpu()

        z500_pred = pred_feat_dict['geopotential'][0, 10//self.downsample_levels, ...].cpu() # b l h w -> h w
        z500_target = target_feat_dict['geopotential'][0, 10//self.downsample_levels, ...].cpu()
        u250_pred = pred_feat_dict['u_component_of_wind'][0, 13//self.downsample_levels, ...].cpu()
        u250_target = target_feat_dict['u_component_of_wind'][0, 13//self.downsample_levels, ...].cpu()
        t850_pred = pred_feat_dict['temperature'][0, 6//self.downsample_levels, ...].cpu()
        t850_target = target_feat_dict['temperature'][0, 6//self.downsample_levels, ...].cpu()
        q850_pred = pred_feat_dict['specific_humidity'][0, 6//self.downsample_levels, ...].cpu()
        q850_target = target_feat_dict['specific_humidity'][0, 6//self.downsample_levels, ...].cpu()

        plot_reconstruction(t2m_pred, # h w
                    t2m_target,
                    f'{self.log_dir}/t2m_{self.current_epoch}.png')
        plot_reconstruction(z500_pred,
                    z500_target,
                    f'{self.log_dir}/z500_{self.current_epoch}.png')
        plot_reconstruction(pr_6h_pred,
                    pr_6h_target,
                    f'{self.log_dir}/PRATEsfc_{self.current_epoch}.png')
        plot_reconstruction(u250_pred,
                    u250_target,
                    f'{self.log_dir}/u250_{self.current_epoch}.png')
        plot_reconstruction(t850_pred,
                    t850_target,
                    f'{self.log_dir}/t850_{self.current_epoch}.png')
        plot_reconstruction(q850_pred,
                    q850_target,
                    f'{self.log_dir}/q850_{self.current_epoch}.png')
        
        plot_spectrum(t2m_pred.unsqueeze(0),
                        t2m_target.unsqueeze(0),
                        f'{self.log_dir}/t2m_spectrum_{self.current_epoch}.png',
                        num_t=1)
        plot_spectrum(z500_pred.unsqueeze(0),
                        z500_target.unsqueeze(0),
                        f'{self.log_dir}/z500_spectrum_{self.current_epoch}.png',
                        num_t=1)
        plot_spectrum(pr_6h_pred.unsqueeze(0),
                        pr_6h_target.unsqueeze(0),
                        f'{self.log_dir}/PRATEsfc_spectrum_{self.current_epoch}.png',
                        num_t=1)
        plot_spectrum(u250_pred.unsqueeze(0),
                        u250_target.unsqueeze(0),
                        f'{self.log_dir}/u250_spectrum_{self.current_epoch}.png',
                        num_t=1)
        plot_spectrum(t850_pred.unsqueeze(0),
                        t850_target.unsqueeze(0),
                        f'{self.log_dir}/t850_spectrum_{self.current_epoch}.png',
                        num_t=1)
        plot_spectrum(q850_pred.unsqueeze(0),
                        q850_target.unsqueeze(0),
                        f'{self.log_dir}/q850_spectrum_{self.current_epoch}.png',
                        num_t=1)
        
    def log_losses(self, loss_dict):
        # calculate the mean loss across batch, shape b for each key, b l for multilevel keys
        t2m_loss = loss_dict['2m_temperature'].mean(0) # surface temp, mean across batch dim
        pr_6h_loss = loss_dict['PRATEsfc'].mean(0) # 6-hour accumulated PRATEsfc
        z500_loss = loss_dict['geopotential'][..., 10//self.downsample_levels].mean(0) # geopotential at level=10
        u250_loss = loss_dict['u_component_of_wind'][..., 13//self.downsample_levels].mean(0) # u wind at level=13
        t850_loss = loss_dict['temperature'][..., 6//self.downsample_levels].mean(0) # temp at level=6
        q850_loss = loss_dict['specific_humidity'][..., 6//self.downsample_levels].mean(0) # specific humidity at level=6
        
        self.log('val/t2m', t2m_loss.item(), on_step=False, on_epoch=True, sync_dist=self.ddp) 
        self.log('val/pr_6h', pr_6h_loss.item(), on_step=False, on_epoch=True, sync_dist=self.ddp)
        self.log('val/z500', z500_loss.item(), on_step=False, on_epoch=True, sync_dist=self.ddp)
        self.log('val/u250', u250_loss.item(), on_step=False, on_epoch=True, sync_dist=self.ddp)
        self.log('val/t850', t850_loss.item(), on_step=False, on_epoch=True, sync_dist=self.ddp)
        self.log('val/q850', q850_loss.item(), on_step=False, on_epoch=True, sync_dist=self.ddp)
    
    def configure_optimizers(self):
        if self.model_name == "AE_Decoder_Only":
            optimizer = torch.optim.Adam(list(self.decoder.parameters()), lr=self.lr)
        else:
            optimizer = torch.optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=self.lr)
            
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

        return [optimizer], [scheduler]
    