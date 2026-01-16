import lightning as L
import torch
from tqdm import tqdm

from modules.models.DCAE import Encoder, Decoder
from common.loss import latitude_weighted_rmse
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

        self.criterion = torch.nn.MSELoss()
        self.n = normalizer

        if self.model_name == "DCAE":
            self.encoder = Encoder(**self.modelconfig["DCAE"]["encoder"])
            self.decoder = Decoder(**self.modelconfig["DCAE"]["decoder"])
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
    
    def compute_loss(self, 
                     surface_pred, surface_target,
                     multilevel_pred, multilevel_target,
                     diagnostic_pred, diagnostic_target):
        
        surface_loss = self.criterion(surface_pred, surface_target)
        multilevel_loss = self.criterion(multilevel_pred, multilevel_target)
        diagnostic_loss = self.criterion(diagnostic_pred, diagnostic_target)

        # can weight this optionally
        return surface_loss + multilevel_loss + diagnostic_loss
    
    def training_step(self, batch, batch_idx):

        surface_data = batch['surface'][:, 0] # b 1 nlat nlon c
        multilevel_data = batch['multilevel'][:, 0] # b 1 nlevel nlat nlon c
        diagnostic_data = batch['diagnostic'][:, 0] # b 1 nlat nlon c

        surface_pred, multilevel_pred, diagnostic_pred = self.forward(surface_data, multilevel_data, diagnostic_data)

        loss = self.compute_loss(surface_pred, surface_data,
                                multilevel_pred, multilevel_data,
                                diagnostic_pred, diagnostic_data)

        self.log("train/loss", loss, on_step=True, on_epoch=True, sync_dist=self.ddp)

        return loss 

    def validation_step(self, batch, batch_idx): 
        
        surface_data = batch['surface'][:, 0] # b 1 nlat nlon c
        multilevel_data = batch['multilevel'][:, 0] # b 1 nlevel nlat nlon c
        diagnostic_data = batch['diagnostic'][:, 0] # b 1 nlat nlon c

        surface_pred, multilevel_pred, diagnostic_pred = self.forward(surface_data, multilevel_data, diagnostic_data)

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

        t2m_pred = pred_feat_dict['2m_temperature'][0].cpu().numpy() #b h w -> h w 
        t2m_target = target_feat_dict['2m_temperature'][0].cpu().numpy()
        pr_6h_pred = pred_feat_dict['PRATEsfc'][0].cpu().numpy()
        pr_6h_target = target_feat_dict['PRATEsfc'][0].cpu().numpy()

        z500_pred = pred_feat_dict['geopotential'][0, 10, ...].cpu().numpy() # b l h w -> h w
        z500_target = target_feat_dict['geopotential'][0, 10, ...].cpu().numpy()
        u250_pred = pred_feat_dict['u_component_of_wind'][0, 13, ...].cpu().numpy()
        u250_target = target_feat_dict['u_component_of_wind'][0, 13, ...].cpu().numpy()
        t850_pred = pred_feat_dict['temperature'][0, 6, ...].cpu().numpy()
        t850_target = target_feat_dict['temperature'][0, 6, ...].cpu().numpy()
        q850_pred = pred_feat_dict['specific_humidity'][0, 6, ...].cpu().numpy()
        q850_target = target_feat_dict['specific_humidity'][0, 6, ...].cpu().numpy()

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
        z500_loss = loss_dict['geopotential'][..., 10].mean(0) # geopotential at level=10
        u250_loss = loss_dict['u_component_of_wind'][..., 13].mean(0) # u wind at level=13
        t850_loss = loss_dict['temperature'][..., 6].mean(0) # temp at level=6
        q850_loss = loss_dict['specific_humidity'][..., 6].mean(0) # specific humidity at level=6
        
        self.log('val/t2m', t2m_loss.item(), on_step=False, on_epoch=True, sync_dist=self.ddp) 
        self.log('val/pr_6h', pr_6h_loss.item(), on_step=False, on_epoch=True, sync_dist=self.ddp)
        self.log('val/z500', z500_loss.item(), on_step=False, on_epoch=True, sync_dist=self.ddp)
        self.log('val/u250', u250_loss.item(), on_step=False, on_epoch=True, sync_dist=self.ddp)
        self.log('val/t850', t850_loss.item(), on_step=False, on_epoch=True, sync_dist=self.ddp)
        self.log('val/q850', q850_loss.item(), on_step=False, on_epoch=True, sync_dist=self.ddp)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), 
                                     lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

        return [optimizer], [scheduler]
    