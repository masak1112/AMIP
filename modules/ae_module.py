import lightning as L
import torch

from common.loss import latitude_weighted_rmse
from common.utils import assemble_input, disassemble_input
from common.plotting import plot_reconstruction, plot_spectrum

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
        self.dataconfig = config['data']
        self.surface_variables = self.dataconfig['surface_variables']
        self.multilevel_variables = self.dataconfig['upper_air_variables']
        self.diagnostic_variables = self.dataconfig['diagnostic_variables']
        self.horizontal_resolution = self.dataconfig['horizontal_resolution']
        self.nlat, self.nlon = self.horizontal_resolution
        self.nlevels = len(self.dataconfig['levels'])

        self.modelconfig = config['model']
        self.model_name = self.modelconfig["model_name"]
        self.lr = self.modelconfig["lr"]
        self.log_dir = config['training']['log_dir']

        self.n = normalizer
   
        if self.model_name == "AE_DIT_DDC":
            from modules.models.DiT import DiT
            from modules.models.AE_simple import BilinearEncoder, BilinearDecoder
            from modules.diffusion.data_dependent_interpolant import DataDependentInterpolant
            self.downsample = BilinearEncoder(**self.modelconfig["AE_DIT_DDC"]["encoder"])
            self.upsample = BilinearDecoder(**self.modelconfig["AE_DIT_DDC"]["encoder"])
            self.decoder = DiT(**self.modelconfig["AE_DIT_DDC"]["decoder"])
            self.scheduler = DataDependentInterpolant(**self.modelconfig["AE_DIT_DDC"]["scheduler"])
            self.decoder_only = True
        else:
            raise NotImplementedError(f"Model {self.model_name} not implemented")

        if config['training']['strategy'] == 'ddp' or config['training']['strategy'] == 'ddp_find_unused_parameters_true':
            self.ddp = True
        else:
            self.ddp = False

        self.save_hyperparameters()

    def forward(self, surface, multilevel, diagnostic):

        # generate latents
        z_surface, z_multilevel, z_diagnostic = self.upsample(*self.downsample(surface, multilevel, diagnostic))
        x = assemble_input(z_surface, z_multilevel, z_diagnostic)

        # diffusion sampling
        y = self.scheduler.sample(x, self.decoder)
        surface_pred, multilevel_pred, diagnostic_pred = disassemble_input(y)

        return surface_pred, multilevel_pred, diagnostic_pred

    
    def training_step(self, batch, batch_idx):
        
        surface_data, multilevel_data, diagnostic_data = batch

        z_surface, z_multilevel, z_diagnostic = self.upsample(*self.downsample(surface_data, multilevel_data, diagnostic_data))
        x = assemble_input(z_surface, z_multilevel, z_diagnostic)
        y = assemble_input(surface_data, multilevel_data, diagnostic_data)
        loss = self.scheduler.compute_loss(x, y, self.decoder)     

        self.log("train/loss", loss, on_step=True, on_epoch=True, sync_dist=self.ddp)

        return loss

    def validation_step(self, batch, batch_idx):
        
        surface_data, multilevel_data, diagnostic_data = batch

        surface_pred, multilevel_pred, diagnostic_pred = self.forward(surface_data, multilevel_data, diagnostic_data)
      
        loss_dict, pred_dict, data_dict = self.compute_loss_val(surface_pred, surface_data,
                                                multilevel_pred, multilevel_data,
                                                diagnostic_pred, diagnostic_data)
        
        self.log_losses(loss_dict)

        if batch_idx == 0: # only plot 1st batch
            if not self.ddp or self.global_rank == 0: # only run plotting on one gpu
                self.save_predictions(pred_dict, data_dict)
        
        #if batch_idx == 0: # only plot 1st batch
        #    if not self.ddp or self.global_rank == 0: # only run plotting on one gpu
        #        self.plot_predictions(pred_dict, data_dict)
    
    @torch.no_grad()
    def compute_loss_val(self,
                surface_pred, surface_data,
                multilevel_pred, multilevel_data,
                diagnostic_pred, diagnostic_data):
        
        surface_pred = self.n.surface_inv_transform(surface_pred)
        multilevel_pred = self.n.upper_air_inv_transform(multilevel_pred)
        diagnostic_pred = self.n.diagnostic_inv_transform(diagnostic_pred)

        surface_data = self.n.surface_inv_transform(surface_data)
        multilevel_data = self.n.upper_air_inv_transform(multilevel_data)
        diagnostic_data = self.n.diagnostic_inv_transform(diagnostic_data)

        pred_feat_dict = {}
        target_feat_dict = {}

        for c, surface_feat_name in enumerate(self.surface_variables):
            pred_feat_dict[surface_feat_name] = surface_pred[:, c] # b nlat nlon
            target_feat_dict[surface_feat_name] = surface_data[:, c]

        for c, multilevel_feat_name in enumerate(self.multilevel_variables):
            pred_feat_dict[multilevel_feat_name] = multilevel_pred[:, c] # b nlevel nlat nlon
            target_feat_dict[multilevel_feat_name] = multilevel_data[:, c]

        for c, diagnostic_feat_name in enumerate(self.diagnostic_variables):
            pred_feat_dict[diagnostic_feat_name] = diagnostic_pred[:, c] # b nlat nlon
            target_feat_dict[diagnostic_feat_name] = diagnostic_data[:, c]

        loss_dict = {k:
                        latitude_weighted_rmse(pred_feat_dict[k], target_feat_dict[k],
                                                nlon=self.nlon, nlat=self.nlat, with_time=False
                                                ) for k in pred_feat_dict.keys()} # b or b l for each key
        
        return loss_dict, pred_feat_dict, target_feat_dict

    def save_predictions(self, pred_feat_dict, target_feat_dict):
        torch.save(pred_feat_dict, f'{self.log_dir}predictions_epoch_{self.current_epoch}.pt')
        torch.save(target_feat_dict, f'{self.log_dir}targets_epoch_{self.current_epoch}.pt')

    def plot_predictions(self, pred_feat_dict, target_feat_dict):

        t2m_pred = pred_feat_dict['2m_temperature'][0].cpu() #b h w -> h w 
        t2m_target = target_feat_dict['2m_temperature'][0].cpu()
        pr_6h_pred = pred_feat_dict['PRATEsfc_24h'][0].cpu()
        pr_6h_target = target_feat_dict['PRATEsfc_24h'][0].cpu()

        z500_pred = pred_feat_dict['geopotential'][0, -10, ...].cpu() # b l h w -> h w
        z500_target = target_feat_dict['geopotential'][0, -10, ...].cpu()
        u250_pred = pred_feat_dict['u_component_of_wind'][0, -13, ...].cpu()
        u250_target = target_feat_dict['u_component_of_wind'][0, -13, ...].cpu()
        t850_pred = pred_feat_dict['temperature'][0, -6, ...].cpu()
        t850_target = target_feat_dict['temperature'][0, -6, ...].cpu()
        q850_pred = pred_feat_dict['specific_total_water'][0, -6, ...].cpu()
        q850_target = target_feat_dict['specific_total_water'][0, -6, ...].cpu()

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
        pr_6h_loss = loss_dict['PRATEsfc_24h'].mean(0) # 6-hour accumulated PRATEsfc
        z500_loss = loss_dict['geopotential'][..., -10].mean(0) # geopotential at level=10
        u250_loss = loss_dict['u_component_of_wind'][..., -13].mean(0) # u wind at level=13
        t850_loss = loss_dict['temperature'][..., -6].mean(0) # temp at level=6
        q850_loss = loss_dict['specific_total_water'][..., -6].mean(0) # specific humidity at level=6
        
        self.log('val/t2m', t2m_loss.item(), on_step=False, on_epoch=True, sync_dist=self.ddp) 
        self.log('val/pr_6h', pr_6h_loss.item(), on_step=False, on_epoch=True, sync_dist=self.ddp)
        self.log('val/z500', z500_loss.item(), on_step=False, on_epoch=True, sync_dist=self.ddp)
        self.log('val/u250', u250_loss.item(), on_step=False, on_epoch=True, sync_dist=self.ddp)
        self.log('val/t850', t850_loss.item(), on_step=False, on_epoch=True, sync_dist=self.ddp)
        self.log('val/q850', q850_loss.item(), on_step=False, on_epoch=True, sync_dist=self.ddp)
    
    def configure_optimizers(self):

        optimizer = torch.optim.Adam(list(self.decoder.parameters()), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

        return [optimizer], [scheduler]
    