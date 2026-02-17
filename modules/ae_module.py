import lightning as L
import torch

from common.loss import latitude_weighted_rmse, WeightedLoss
from common.utils import assemble_input, disassemble_input
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
        self.separate_diagnostic = config['model'].get('separate_diagnostic', False)
        self.surface_variable_weight = config['model'].get('surface_variable_weight', None)
        self.multi_level_variable_weight = config['model'].get('multi_level_variable_weight', None)
        self.diag_variable_weight = config['model'].get('diag_variable_weight', None)
        self.level_weight = config['model'].get('level_weight', "equal")
        self.spectral_loss_weight = config['model'].get('spectral_loss_weight', 0.0)

        self.criterion = WeightedLoss(latitude_resolution=180,
                                      longitude_resolution=360,
                                      nlevels = 26 // self.downsample_levels,
                                      level_weight=self.level_weight,
                                      surface_variable_weight=self.surface_variable_weight,
                                      multi_level_variable_weight=self.multi_level_variable_weight,
                                      diag_variable_weight=self.diag_variable_weight)
        
        self.spectral_criterion = None
        if self.spectral_loss_weight > 0.0:
            from common.loss import SpectralBaseLoss
            self.spectral_criterion = SpectralBaseLoss(img_shape=(180, 360))

        self.n = normalizer

        self.history = False
        self.decoder_only = False
        self.stochastic = False
        self.n_ensemble = 1
        self.scheduler = None
        
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
            self.decoder_only = True
        elif self.model_name == "AE_Atlas":
            from modules.models.AE_simple import BilinearEncoder
            from modules.models.AE_attn import NattenCombineDiT
            self.encoder = BilinearEncoder(**self.modelconfig["AE_Atlas"]["encoder"])
            self.decoder = NattenCombineDiT(**self.modelconfig["AE_Atlas"]["decoder"])
            self.history = True
            self.decoder_only = True
        elif self.model_name == "AE_ClimaDiT":
            from modules.models.AE_dit import ClimaDiT
            from modules.models.AE_simple import BilinearEncoder
            self.encoder = BilinearEncoder(**self.modelconfig["AE_ClimaDiT"]["encoder"])
            self.decoder = ClimaDiT(**self.modelconfig["AE_ClimaDiT"]["decoder"])
            self.history = True
            self.decoder_only = True
        elif self.model_name == "AE_History":
            from modules.models.AE_simple import DecoderHistory
            from modules.models.AE_simple import BilinearEncoder
            self.encoder = BilinearEncoder(**self.modelconfig["AE_History"]["encoder"])
            self.decoder = DecoderHistory(**self.modelconfig["AE_History"]["decoder"])
            self.history = True
            self.decoder_only = True
        elif self.model_name == "AE_History2":
            from modules.models.AE_decoder import DecoderHistory
            from modules.models.AE_simple import BilinearEncoder
            self.encoder = BilinearEncoder(**self.modelconfig["AE_History"]["encoder"])
            self.decoder = DecoderHistory(**self.modelconfig["AE_History"]["decoder"])
            self.history = True
            self.decoder_only = True
        elif self.model_name == "AE_SI":
            from modules.models.AE_Unet import DecoderUnet
            from modules.models.AE_simple import BilinearEncoder, BilinearDecoder
            from modules.diffusion.interpolant import DriftScheduler
            self.downsample = BilinearEncoder(**self.modelconfig["AE_SI"]["encoder"])
            self.upsample = BilinearDecoder(**self.modelconfig["AE_SI"]["encoder"])

            self.decoder = DecoderUnet(**self.modelconfig["AE_SI"]["decoder"])
            self.scheduler = DriftScheduler(**self.modelconfig["AE_SI"]["scheduler"])
            self.decoder_only = True 
        elif self.model_name == "AE_SIT":
            from modules.models.AE_dit import ClimaSiT
            from modules.models.AE_simple import BilinearEncoder, BilinearDecoder
            from modules.diffusion.interpolant import DriftScheduler
            self.downsample = BilinearEncoder(**self.modelconfig["AE_SI"]["encoder"])
            self.upsample = BilinearDecoder(**self.modelconfig["AE_SI"]["encoder"])

            self.decoder = ClimaSiT(**self.modelconfig["AE_SI"]["decoder"])
            self.scheduler = DriftScheduler(**self.modelconfig["AE_SI"]["scheduler"])
            self.decoder_only = True
        elif self.model_name == "AE_SI_DDC":
            from modules.models.SI_DiT import SIDiT
            from modules.models.AE_simple import BilinearEncoder, BilinearDecoder
            from modules.diffusion.data_dependent_interpolant import DataDependentInterpolant
            self.downsample = BilinearEncoder(**self.modelconfig["AE_SI_DDC"]["encoder"])
            self.upsample = BilinearDecoder(**self.modelconfig["AE_SI_DDC"]["encoder"])
            self.decoder = SIDiT(**self.modelconfig["AE_SI_DDC"]["decoder"])
            self.scheduler = DataDependentInterpolant(**self.modelconfig["AE_SI_DDC"]["scheduler"])
            self.history = self.modelconfig["AE_SI_DDC"]["decoder"].get("use_history", False)
            self.decoder_only = True
        elif self.model_name == "AE_Flow":
            from modules.models.SI_DiT import SIDiT
            from modules.models.AE_simple import BilinearEncoder, BilinearDecoder
            from modules.diffusion.flow_matching import ConditionalFlowMatching
            self.downsample = BilinearEncoder(**self.modelconfig["AE_Flow"]["encoder"])
            self.upsample = BilinearDecoder(**self.modelconfig["AE_Flow"]["encoder"])
            self.decoder = SIDiT(**self.modelconfig["AE_Flow"]["decoder"])
            self.scheduler = ConditionalFlowMatching(**self.modelconfig["AE_Flow"]["scheduler"])
            self.history = self.modelconfig["AE_Flow"]["decoder"].get("use_history", False)
            self.decoder_only = True
        elif self.model_name == "AE_Stochastic":
            from modules.models.AE_decoder import StochasticDecoderHistory
            from modules.models.AE_simple import BilinearEncoder
            from common.loss import SpectralCRPSLoss, FairCRPSLoss
            self.encoder = BilinearEncoder(**self.modelconfig["AE_Stochastic"]["encoder"])
            self.decoder = StochasticDecoderHistory(**self.modelconfig["AE_Stochastic"]["decoder"])
            self.history = True
            self.decoder_only = True
            self.stochastic = True
            self.n_ensemble = config['model'].get('n_ensemble', 2)
            # Replace criterion with FairCRPSLoss (almost-fair when alpha < 1)
            self.crps_alpha = config['model'].get('crps_alpha', 1.0)
            self.criterion = FairCRPSLoss(
                latitude_resolution=180,
                longitude_resolution=360,
                nlevels=26 // self.downsample_levels,
                level_weight=self.level_weight,
                surface_variable_weight=self.surface_variable_weight,
                multi_level_variable_weight=self.multi_level_variable_weight,
                diag_variable_weight=self.diag_variable_weight,
                alpha=self.crps_alpha,
                n_ensemble=self.n_ensemble,
            )
            self.spectral_criterion = SpectralCRPSLoss(img_shape=(180, 360),
                                                       alpha = self.crps_alpha)

        else:
            raise NotImplementedError(f"Model {self.model_name} not implemented")

        if config['training']['strategy'] == 'ddp' or config['training']['strategy'] == 'ddp_find_unused_parameters_true':
            self.ddp = True
        else:
            self.ddp = False

        self.save_hyperparameters()

    def forward(self, surface, multilevel, diagnostic):
        if self.scheduler is not None:
            z_surface, z_multilevel, z_diagnostic = self.upsample(*self.downsample(surface, multilevel, diagnostic))
            x = assemble_input(z_surface, z_multilevel, z_diagnostic)
            y = self.scheduler.sample(x, self.decoder)
            surface_pred, multilevel_pred, diagnostic_pred = disassemble_input(y)

        elif self.separate_diagnostic:
            multilevel = multilevel[..., :5] # remove cloud and vertical velocity from inputs
            z_surface, z_multilevel, _ = self.encoder(surface, multilevel, None)
            surface_pred, multilevel_pred, diagnostic_pred = self.decoder(z_surface, z_multilevel, None)
        else:
            z_surface, z_multilevel, z_diagnostic = self.encoder(surface, multilevel, diagnostic)
            surface_pred, multilevel_pred, diagnostic_pred = self.decoder(z_surface, z_multilevel, z_diagnostic)

        return surface_pred, multilevel_pred, diagnostic_pred
    
    def forward_history(self, surface_history, multilevel_history, diagnostic_history,
                        surface, multilevel, diagnostic):

        if self.scheduler is not None:
            z_surface, z_multilevel, z_diagnostic = self.upsample(*self.downsample(surface, multilevel, diagnostic))
            x = assemble_input(z_surface, z_multilevel, z_diagnostic)
            cond = assemble_input(surface_history, multilevel_history, diagnostic_history)
            y = self.scheduler.sample(x, self.decoder, cond=cond)
            surface_pred, multilevel_pred, diagnostic_pred = disassemble_input(y)

        elif self.separate_diagnostic:
            multilevel = multilevel[..., :5] # remove cloud and vertical velocity from inputs
            multilevel_history = multilevel_history[..., :5] 
            z_surface, z_multilevel, _ = self.encoder(surface, multilevel, None)
            surface_pred, multilevel_pred, diagnostic_pred = self.decoder(surface_history, multilevel_history, None,
                                                                z_surface, z_multilevel, None)
        else:
            z_surface, z_multilevel, z_diagnostic = self.encoder(surface, multilevel, diagnostic)
            surface_pred, multilevel_pred, diagnostic_pred = self.decoder(
                surface_history, multilevel_history, diagnostic_history,
                z_surface, z_multilevel, z_diagnostic)

        return surface_pred, multilevel_pred, diagnostic_pred
    
    def compute_loss(self, 
                     surface_pred, surface_target,
                     multilevel_pred, multilevel_target,
                     diagnostic_pred, diagnostic_target):
        
        if self.spectral_criterion is None:
            return self.criterion(surface_pred, surface_target,
                        multilevel_pred, multilevel_target,
                        diagnostic_pred, diagnostic_target)
        else:
            mse_loss = self.criterion(surface_pred, surface_target,
                        multilevel_pred, multilevel_target,
                        diagnostic_pred, diagnostic_target)
            
            x_pred = assemble_input(surface_pred, multilevel_pred, diagnostic_pred)
            x_target = assemble_input(surface_target, multilevel_target, diagnostic_target)

            spectral_loss = self.spectral_criterion(x_pred, x_target)

            return mse_loss + self.spectral_loss_weight * spectral_loss
    
    
    def training_step(self, batch, batch_idx):
        
        # diffusion training
        if self.scheduler is not None:
            if not self.history:
                surface_data = batch['surface'][:, 0] # b nlat nlon c
                multilevel_data = batch['multilevel'][:, 0] # b nlevel nlat nlon c
                diagnostic_data = batch['diagnostic'][:, 0] # b nlat nlon c

                z_surface, z_multilevel, z_diagnostic = self.upsample(*self.downsample(surface_data, multilevel_data, diagnostic_data))
                x = assemble_input(z_surface, z_multilevel, z_diagnostic)
                y = assemble_input(surface_data, multilevel_data, diagnostic_data)
                loss = self.scheduler.compute_loss(x, y, self.decoder)

            else:
                surface_history = batch['surface'][:, 0] # b nlat nlon c
                multilevel_history = batch['multilevel'][:, 0]
                diagnostic_history = batch['diagnostic'][:, 0]

                surface_data = batch['surface'][:, 1] # b nlat nlon c
                multilevel_data = batch['multilevel'][:, 1]
                diagnostic_data = batch['diagnostic'][:, 1]

                cond = assemble_input(surface_history, multilevel_history, diagnostic_history)
                z_surface, z_multilevel, z_diagnostic = self.upsample(*self.downsample(surface_data, multilevel_data, diagnostic_data))
                x = assemble_input(z_surface, z_multilevel, z_diagnostic)
                y = assemble_input(surface_data, multilevel_data, diagnostic_data)

                loss = self.scheduler.compute_loss(x, y, self.decoder, cond=cond)

        # standard autoencoder training
        elif not self.history:
            surface_data = batch['surface'][:, 0] # b nlat nlon c
            multilevel_data = batch['multilevel'][:, 0] # b nlevel nlat nlon c
            diagnostic_data = batch['diagnostic'][:, 0] # b nlat nlon c

            surface_pred, multilevel_pred, diagnostic_pred = self.forward(surface_data, multilevel_data, diagnostic_data)

        # history-based training for decoder-only
        else:
            surface_history = batch['surface'][:, 0] # b nlat nlon c
            multilevel_history = batch['multilevel'][:, 0]
            diagnostic_history = batch['diagnostic'][:, 0]

            surface_data = batch['surface'][:, 1] # b nlat nlon c
            multilevel_data = batch['multilevel'][:, 1]
            diagnostic_data = batch['diagnostic'][:, 1]

            if self.stochastic:
                s_pred1, m_pred1, d_pred1 = self.forward_history(
                    surface_history, multilevel_history, diagnostic_history,
                    surface_data, multilevel_data, diagnostic_data)
                s_pred2, m_pred2, d_pred2 = self.forward_history(
                    surface_history, multilevel_history, diagnostic_history,
                    surface_data, multilevel_data, diagnostic_data)

                # Fair CRPS loss
                crps_loss = self.criterion(s_pred1, s_pred2, surface_data,
                                        m_pred1, m_pred2, multilevel_data,
                                        d_pred1, d_pred2, diagnostic_data)

                if self.spectral_criterion is not None:
                    forecasts = torch.stack([assemble_input(s_pred1, m_pred1, d_pred1),
                                            assemble_input(s_pred2, m_pred2, d_pred2)], dim=1) # b n_ensemble c h w
                    observation = assemble_input(surface_data, multilevel_data, diagnostic_data) # b c h w
                    spectral_loss = self.spectral_loss_weight * self.spectral_criterion(forecasts, observation)
                    loss = crps_loss + spectral_loss
                    self.log("train/spectral_loss", spectral_loss, on_step=True, on_epoch=True, sync_dist=self.ddp)
                else:
                    loss = crps_loss

                self.log("train/crps_loss", crps_loss, on_step=True, on_epoch=True, sync_dist=self.ddp)

            else:
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
        if self.decoder_only:
            optimizer = torch.optim.Adam(list(self.decoder.parameters()), lr=self.lr)
        else:
            optimizer = torch.optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=self.lr)
            
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

        return [optimizer], [scheduler]
    