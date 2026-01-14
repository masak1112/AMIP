import lightning as L
import torch
from tqdm import tqdm

from modules.models.DiT import ArchesDiT
from modules.diffusion.flow_matching import FlowScheduler
from common.loss import latitude_weighted_rmse
from common.plotting import plot_result, plot_spectrum, plot_bias
from data.amip import ClimatologyLoader, SURFACE_VARIABLES, MULTILEVEL_VARIABLES, DIAGNOSTIC_VARIABLES
from data.normalizer import Normalizer

class TrainModule(L.LightningModule):
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

        dataconfig = config['data'] 
        self.climatology_loader = ClimatologyLoader(data_path=dataconfig["train_data_path"],
                                    norm_stats_path=dataconfig["norm_stats_path"],
                                    climatology_path = dataconfig["climatology_path"],
                                    horizon=dataconfig['climatology_horizon'],
                                    start_time=dataconfig['climatology_start'],)

        self.criterion = torch.nn.MSELoss()
        self.n = normalizer

        if self.model_name == "sfno":
            from modules.models.SFNO import SphericalFourierNeuralOperatorNet
            self.model = SphericalFourierNeuralOperatorNet(params={},
                                                           **self.modelconfig["sfno"])
            self.diffusion=False 
        elif self.model_name == 'flow':
            self.model = ArchesDiT(**self.modelconfig["dit"])
            self.scheduler = FlowScheduler(**self.modelconfig["flow"])
            self.diagnostic_channels = self.modelconfig["dit"]['encode_decode_params']['diagnostic_ch']
            self.diffusion=True 
        else:
            raise NotImplementedError(f"Model {self.model_name} not implemented")

        if config['training']['strategy'] == 'ddp' or config['training']['strategy'] == 'ddp_find_unused_parameters_true':
            self.ddp = True
        else:
            self.ddp = False

        self.save_hyperparameters()

    def forward(self, surface, multilevel, forcing, invariant, scalars):
        if self.diffusion:
            surface_pred, multilevel_pred, diagnostic_pred = self.scheduler.sample(self.model, surface, multilevel, forcing, invariant, 
                                                                                   scalars, self.diagnostic_channels)
        else: # directly predict
            surface_pred, multilevel_pred, diagnostic_pred = self.model(surface, multilevel, forcing, invariant, scalars)

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

        surface_data = batch['surface'] # b t nlat nlon c
        multilevel_data = batch['multilevel'] # b t nlat nlon nlevel c
        forcing_data = batch['forcing'] # b t nlat nlon c
        invariant_input = batch['invariant'] # b nlat nlon c
        scalar_data = batch['scalars'] # b t 2
        diagnostic_data = batch['diagnostic'] # b t nlat nlon c

        surface_input = surface_data[:, 0] # b nlat nlon c
        multilevel_input = multilevel_data[:, 0] # b nlat nlon nlevel c
        forcing_input = forcing_data[:, 0] # b nlat nlon c
        scalar_input = scalar_data[:, 0] # b 2

        surface_target = surface_data[:, 1] # b nlat nlon c
        multilevel_target = multilevel_data[:, 1] # b nlat nlon nlevel c
        diagnostic_target = diagnostic_data[:, 1] # b nlat nlon c

        if self.diffusion:
            loss = self.scheduler.compute_loss(self.model,
                                               surface_input,
                                               multilevel_input,
                                               forcing_input,
                                               invariant_input,
                                               scalar_input,
                                               surface_target,
                                               multilevel_target,
                                               diagnostic_target)   
        else:
            surface_pred, multilevel_pred, diagnostic_pred \
                = self.forward(surface_input,
                            multilevel_input,
                            forcing_input,
                            invariant_input,
                            scalar_input,) 


            loss = self.compute_loss(surface_pred, surface_target,
                                    multilevel_pred, multilevel_target,
                                    diagnostic_pred, diagnostic_target)

        self.log("train/loss", loss, on_step=True, on_epoch=True, sync_dist=self.ddp)

        return loss 

    def validation_step(self, batch, batch_idx): 
    
        loss_dict, pred_feat_dict, target_feat_dict = self.predict(batch)
        self.log_losses(loss_dict)

        # visualize the prediction for first batch and on one gpu
        if batch_idx == 0:
            if self.ddp and self.global_rank != 0:
                pass
            else: 
                self.plot_predicitons(pred_feat_dict, target_feat_dict)
                batch_climatology = self.climatology_loader.get_data(device=batch['surface'].device)   
                bias_loss_dict, pred_bias = self.predict_bias(batch_climatology)
    
    @torch.no_grad()
    def predict(self, batch):
        surface_data = batch['surface'] # b t nlat nlon c
        multilevel_data = batch['multilevel'] # b t nlat nlon nlevel c
        forcing_data = batch['forcing'] # b t nlat nlon c
        invariant_input = batch['invariants'] # b nlat nlon c
        scalar_data = batch['scalars'] # b t 2
        diagnostic_data = batch['diagnostic'] # b t nlat nlon c
                
        surface_input = surface_data[:, 0] # b nlat nlon c
        multilevel_input = multilevel_data[:, 0] # b nlat nlon nlevel c

        surface_target = surface_data[:, 1:] # b t nlat nlon c
        multilevel_target = multilevel_data[:, 1:] # b t nlat nlon nlevel c
        diagnostic_target = diagnostic_data[:, 1:] # b t nlat nlon c

        # TODO: optimize memory usage by calculating losses on the fly. Only plot certain timesteps, levels, variables of interest.

        surface_pred_all = torch.zeros_like(surface_target, device=surface_data.device) # b t nlat nlon c
        multilevel_pred_all = torch.zeros_like(multilevel_target, device=multilevel_data.device) # b t nlat nlon nlevel c
        diagnostic_pred_all = torch.zeros_like(diagnostic_target, device=diagnostic_data.device) # b t nlat nlon c

        for t in range(surface_target.shape[1]):
            # assemble forcings
            forcing_input = forcing_data[:, t] # b nlat nlon c
            scalar_input = scalar_data[:, t] # b 2
            
            # make prediction
            surface_pred, multilevel_pred, diagnostic_pred \
                = self.forward(surface_input,
                            multilevel_input,
                            forcing_input,
                            invariant_input,
                            scalar_input,)

            # save prediction
            surface_pred_all[:, t] = surface_pred
            multilevel_pred_all[:, t] =  multilevel_pred
            diagnostic_pred_all[:, t] = diagnostic_pred

            # update inputs
            surface_input = surface_pred
            multilevel_input = multilevel_pred

        # denormalize
        surface_pred_all = self.n.denormalize_surface(surface_pred_all)
        multilevel_pred_all = self.n.denormalize_multilevel(multilevel_pred_all)
        diagnostic_pred_all = self.n.denormalize_diagnostic(diagnostic_pred_all)
        surface_target = self.n.denormalize_surface(surface_target)
        multilevel_target = self.n.denormalize_multilevel(multilevel_target)
        diagnostic_target = self.n.denormalize_diagnostic(diagnostic_target)

        pred_feat_dict = {}
        target_feat_dict = {}

        for c, surface_feat_name in enumerate(SURFACE_VARIABLES):
            pred_feat_dict[surface_feat_name] = surface_pred_all[..., c] # b t nlat nlon
            target_feat_dict[surface_feat_name] = surface_target[..., c]

        for c, multilevel_feat_name in enumerate(MULTILEVEL_VARIABLES):
            pred_feat_dict[multilevel_feat_name] = multilevel_pred_all[..., c] # b t nlevel nlat nlon 
            target_feat_dict[multilevel_feat_name] = multilevel_target[..., c]

        for c, diagnostic_feat_name in enumerate(DIAGNOSTIC_VARIABLES):
            pred_feat_dict[diagnostic_feat_name] = diagnostic_pred_all[..., c] # b t nlat nlon
            target_feat_dict[diagnostic_feat_name] = diagnostic_target[..., c]

        nlat, nlon = surface_data.shape[2], surface_data.shape[3]
        loss_dict = {k:
                        latitude_weighted_rmse(pred_feat_dict[k], target_feat_dict[k],
                                                nlon=nlon, nlat=nlat,
                                                ) for k in pred_feat_dict.keys()} # b t or b t l for each key
        
        return loss_dict, pred_feat_dict, target_feat_dict
        
    @torch.no_grad()
    def predict_bias(self, batch):
        # b = 1 
        # assume these are normalized
        surface_input = batch['surface'] # b nlat nlon c
        multilevel_input = batch['multilevel'] # b nlat nlon nlevel c
        forcing_data = batch['forcing'] # b t nlat nlon c
        invariant_input = batch['invariants'] # b nlat nlon c
        diagnostic_data = batch['diagnostic'] # b nlat nlon c
        scalar_data = batch['scalars'] # b t 2
        bias_dict = batch['climatology'] # dict of nlat nlon or nlat nlon nlevel tensors

        horizon = forcing_data.shape[1]

        # keep track of unnormalized running totals
        running_total_surface = self.n.denormalize_surface(surface_input.clone())
        running_total_multilevel = self.n.denormalize_surface(multilevel_input.clone())
        running_total_diagnostic = self.n.denormalize_surface(diagnostic_data)

        num = 1

        for i in tqdm(range(horizon), leave=False):
            forcing_input = forcing_data[:, i] # b nlat nlon c
            scalar_input = scalar_data[:, i] # b 2

            surface_pred, multilevel_pred, diagnostic_pred \
                = self.forward(surface_input,
                            multilevel_input,
                            forcing_input,
                            invariant_input,
                            scalar_input,)

            running_total_surface += self.n.denormalize_surface(surface_pred)
            running_total_multilevel += self.n.denormalize_multilevel(multilevel_pred)
            running_total_diagnostic += self.n.denormalize_diagnostic(diagnostic_pred)
            num += 1
            
            surface_input = surface_pred
            multilevel_input = multilevel_pred

        surface_bias = running_total_surface / num # b nlat nlon c
        multilevel_bias = running_total_multilevel / num # b nlat nlon nlevel c
        diagnostic_bias = running_total_diagnostic / num # b nlat nlon c

        pred_feat_dict = {}
        for c, surface_feat_name in enumerate(SURFACE_VARIABLES):
            pred_feat_dict[surface_feat_name] = surface_bias[..., c] # b nlat nlon 

        for c, multilevel_feat_name in enumerate(MULTILEVEL_VARIABLES):
            pred_feat_dict[multilevel_feat_name] = multilevel_bias[..., c] # b nlat nlon nlevel 

        for c, diagnostic_feat_name in enumerate(DIAGNOSTIC_VARIABLES):
            pred_feat_dict[diagnostic_feat_name] = diagnostic_bias[..., c] # b nlat nlon 

        bias_key_list = ['2m_temperature', 'PRATEsfc', 'geopotential', 'temperature', 'u_component_of_wind', 'specific_humidity']
        result_dict = {}

        for var_name in bias_key_list:
            bias = bias_dict[var_name].unsqueeze(0) # 1 nlat nlon or 1 nlat nlon nlevel
            pred_k = pred_feat_dict[var_name] # 1 nlat nlon or 1 nlat nlon nlevel
            
            l = -1 
            if var_name == "geopotential":
                l = 10
            elif var_name == "u_component_of_wind":
                l = 13
            elif var_name == "temperature" or var_name == "specific_humidity":
                l = 6
            
            if l != -1:
                pred_k = pred_feat_dict[..., l]
                bias = bias[..., l]

            nlat, nlon = bias.shape[1], bias.shape[2]
            loss = latitude_weighted_rmse(pred_k, 
                                        bias,
                                        nlon=nlon,
                                        nlat=nlat,
                                        with_time=False)
            
            result_dict[var_name] = loss.item()
            plot_bias(pred_k[0].cpu(), bias[0].cpu(), save_path=f"{self.log_dir}/{var_name}_bias_{self.current_epoch}.png")

        self.log('bias/t2m', result_dict['2m_temperature'], on_step=False, on_epoch=True, sync_dist=self.ddp)
        self.log('bias/pr_6h', result_dict['PRATEsfc'], on_step=False, on_epoch=True, sync_dist=self.ddp)
        self.log('bias/z500', result_dict['geopotential'], on_step=False, on_epoch=True, sync_dist=self.ddp)
        self.log('bias/u250', result_dict['u_component_of_wind'], on_step=False, on_epoch=True, sync_dist=self.ddp)
        self.log('bias/t850', result_dict['temperature'], on_step=False, on_epoch=True, sync_dist=self.ddp)
        self.log('bias/q850', result_dict['specific_humidity'], on_step=False, on_epoch=True, sync_dist=self.ddp)

        return result_dict, pred_feat_dict
    
    def plot_predicitons(self, pred_feat_dict, target_feat_dict):

        t2m_pred = pred_feat_dict['2m_temperature'][0].cpu().numpy() #b t h w -> t h w 
        t2m_target = target_feat_dict['2m_temperature'][0].cpu().numpy()
        z500_pred = pred_feat_dict['geopotential'][0, :, 10, ...].cpu().numpy() # b t l h w -> t h w
        z500_target = target_feat_dict['geopotential'][0, :, 10, ...].cpu().numpy()
        pr_6h_pred = pred_feat_dict['PRATEsfc'][0].cpu().numpy()
        pr_6h_target = target_feat_dict['PRATEsfc'][0].cpu().numpy()
        u250_pred = pred_feat_dict['u_component_of_wind'][0, :, 13, ...].cpu().numpy()
        u250_target = target_feat_dict['u_component_of_wind'][0, :, 13, ...].cpu().numpy()
        t850_pred = pred_feat_dict['temperature'][0, :, 6, ...].cpu().numpy()
        t850_target = target_feat_dict['temperature'][0, :, 6, ...].cpu().numpy()
        q850_pred = pred_feat_dict['specific_humidity'][0, :, 6, ...].cpu().numpy()
        q850_target = target_feat_dict['specific_humidity'][0, :, 6, ...].cpu().numpy()

        plot_result(t2m_pred, # t h w
                    t2m_target,
                    f'{self.log_dir}/t2m_{self.current_epoch}.png')
        plot_result(z500_pred,
                    z500_target,
                    f'{self.log_dir}/z500_{self.current_epoch}.png')
        plot_result(pr_6h_pred,
                    pr_6h_target,
                    f'{self.log_dir}/PRATEsfc_{self.current_epoch}.png')
        plot_result(u250_pred,
                    u250_target,
                    f'{self.log_dir}/u250_{self.current_epoch}.png')
        plot_result(t850_pred,
                    t850_target,
                    f'{self.log_dir}/t850_{self.current_epoch}.png')
        plot_result(q850_pred,
                    q850_target,
                    f'{self.log_dir}/q850_{self.current_epoch}.png')
        
        plot_spectrum(t2m_pred,
                        t2m_target,
                        f'{self.log_dir}/t2m_spectrum_{self.current_epoch}.png')
        plot_spectrum(z500_pred,
                        z500_target,
                        f'{self.log_dir}/z500_spectrum_{self.current_epoch}.png')
        plot_spectrum(pr_6h_pred,
                        pr_6h_target,
                        f'{self.log_dir}/PRATEsfc_spectrum_{self.current_epoch}.png')
        plot_spectrum(u250_pred,
                        u250_target,
                        f'{self.log_dir}/u250_spectrum_{self.current_epoch}.png')
        plot_spectrum(t850_pred,
                        t850_target,
                        f'{self.log_dir}/t850_spectrum_{self.current_epoch}.png')
        plot_spectrum(q850_pred,
                        q850_target,
                        f'{self.log_dir}/q850_spectrum_{self.current_epoch}.png')
        
    def log_losses(self, loss_dict):
        # calculate the mean loss across batch, shape b t for each key, b t l for multilevel keys
        t2m_loss = loss_dict['2m_temperature'].mean(0) # surface temp, mean across batch dim
        pr_6h_loss = loss_dict['PRATEsfc'].mean(0) # 6-hour accumulated PRATEsfc
        z500_loss = loss_dict['geopotential'][..., 10].mean(0) # geopotential at level=10
        u250_loss = loss_dict['u_component_of_wind'][..., 13].mean(0) # u wind at level=13
        t850_loss = loss_dict['temperature'][..., 6].mean(0) # temp at level=6
        q850_loss = loss_dict['specific_humidity'][..., 6].mean(0) # specific humidity at level=6
        
        self.log('val/t2m_6', t2m_loss[0].item(), on_step=False, on_epoch=True, sync_dist=self.ddp) # 6 hours
        self.log('val/t2m_24', t2m_loss[3].item(), on_step=False, on_epoch=True, sync_dist=self.ddp) # 1 day
        self.log('val/t2m_72', t2m_loss[11].item(), on_step=False, on_epoch=True, sync_dist=self.ddp) # 3 day
        self.log('val/t2m_120', t2m_loss[19].item(), on_step=False, on_epoch=True, sync_dist=self.ddp) # 5 day
        #self.log('val/t2m_240', t2m_loss[39].item(), on_step=False, on_epoch=True, sync_dist=self.ddp) # 10 day

        self.log('val/pr_6h_6', pr_6h_loss[0].item(), on_step=False, on_epoch=True, sync_dist=self.ddp)
        self.log('val/pr_6h_24', pr_6h_loss[3].item(), on_step=False, on_epoch=True, sync_dist=self.ddp)
        self.log('val/pr_6h_72', pr_6h_loss[11].item(), on_step=False, on_epoch=True, sync_dist=self.ddp)
        self.log('val/pr_6h_120', pr_6h_loss[19].item(), on_step=False, on_epoch=True, sync_dist=self.ddp)
        #self.log('val/pr_6h_240', pr_6h_loss[39].item(), on_step=False, on_epoch=True, sync_dist=self.ddp)

        self.log('val/z500_6', z500_loss[0].item(), on_step=False, on_epoch=True, sync_dist=self.ddp)
        self.log('val/z500_24', z500_loss[3].item(), on_step=False, on_epoch=True, sync_dist=self.ddp)
        self.log('val/z500_72', z500_loss[11].item(), on_step=False, on_epoch=True, sync_dist=self.ddp)
        self.log('val/z500_120', z500_loss[19].item(), on_step=False, on_epoch=True, sync_dist=self.ddp)
        #self.log('val/z500_240', z500_loss[39].item(), on_step=False, on_epoch=True, sync_dist=self.ddp)

        self.log('val/u250_6', u250_loss[0].item(), on_step=False, on_epoch=True, sync_dist=self.ddp)
        self.log('val/u250_24', u250_loss[3].item(), on_step=False, on_epoch=True, sync_dist=self.ddp)
        self.log('val/u250_72', u250_loss[11].item(), on_step=False, on_epoch=True, sync_dist=self.ddp)
        self.log('val/u250_120', u250_loss[19].item(), on_step=False, on_epoch=True, sync_dist=self.ddp)
        #self.log('val/u250_240', u250_loss[39].item(), on_step=False, on_epoch=True, sync_dist=self.ddp)

        self.log('val/t850_6', t850_loss[0].item(), on_step=False, on_epoch=True, sync_dist=self.ddp)
        self.log('val/t850_24', t850_loss[3].item(), on_step=False, on_epoch=True, sync_dist=self.ddp)
        self.log('val/t850_72', t850_loss[11].item(), on_step=False, on_epoch=True, sync_dist=self.ddp)
        self.log('val/t850_120', t850_loss[19].item(), on_step=False, on_epoch=True, sync_dist=self.ddp)
        #self.log('val/t850_240', t850_loss[39].item(), on_step=False, on_epoch=True, sync_dist=self.ddp)

        self.log('val/q850_6', q850_loss[0].item(), on_step=False, on_epoch=True, sync_dist=self.ddp)
        self.log('val/q850_24', q850_loss[3].item(), on_step=False, on_epoch=True, sync_dist=self.ddp)
        self.log('val/q850_72', q850_loss[11].item(), on_step=False, on_epoch=True, sync_dist=self.ddp)
        self.log('val/q850_120', q850_loss[19].item(), on_step=False, on_epoch=True, sync_dist=self.ddp)
        #self.log('val/q850_240', q850_loss[39].item(), on_step=False, on_epoch=True, sync_dist=self.ddp)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

        return [optimizer], [scheduler]
    