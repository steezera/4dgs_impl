class GaussianModel:
    def __init__(self, sh_degree: int, gaussian_dim: int = 3, time_duration: list = [-0.5, 0.5],
                 rot_4d: bool = False, force_sh_3d: bool = False, sh_degree_t: int = 0):
        # 기존 4D 파라미터 초기화 (예시)
        self.gaussian_dim = gaussian_dim
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0

        self._t = torch.empty(0)
        self._scaling_t = torch.empty(0)
        self.time_duration = time_duration
        self.rot_4d = rot_4d
        self._rotation_r = torch.empty(0)
        self.force_sh_3d = force_sh_3d
        self.t_gradient_accum = torch.empty(0)
        self.env_map = torch.empty(0)

        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self.active_sh_degree_t = 0
        self.max_sh_degree_t = sh_degree_t

        self.setup_functions()

        # ===== 추가: 정적 Gaussian 파라미터 초기화 =====
        self.static_xyz = torch.empty(0)
        self.static_features_dc = torch.empty(0)
        self.static_features_rest = torch.empty(0)
        self.static_scaling = torch.empty(0)
        self.static_rotation = torch.empty(0)
        self.static_opacity = torch.empty(0)
        self.static_max_radii2D = torch.empty(0)
        self.static_xyz_gradient_accum = torch.empty(0)
        # =================================================

def capture(self):
    if self.gaussian_dim == 4:
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.t_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self._t,
            self._scaling_t,
            self._rotation_r,
            self.rot_4d,
            self.env_map,
            self.active_sh_degree_t,
            # ===== 추가: 정적 파라미터 =====
            self.static_xyz,
            self.static_features_dc,
            self.static_features_rest,
            self.static_scaling,
            self.static_rotation,
            self.static_opacity,
            self.static_max_radii2D,
            self.static_xyz_gradient_accum
            # =================================
        )
    else:
        # 3D인 경우는 기존 방식 사용
        pass

def restore(self, model_args, training_args):
    if self.gaussian_dim == 4:
        (self.active_sh_degree,
         self._xyz,
         self._features_dc,
         self._features_rest,
         self._scaling,
         self._rotation,
         self._opacity,
         self.max_radii2D,
         self.xyz_gradient_accum,
         self.t_gradient_accum,
         self.denom,
         opt_dict,
         self.spatial_lr_scale,
         self._t,
         self._scaling_t,
         self._rotation_r,
         self.rot_4d,
         self.env_map,
         self.active_sh_degree_t,
         # ===== 추가: 정적 파라미터 복원 =====
         self.static_xyz,
         self.static_features_dc,
         self.static_features_rest,
         self.static_scaling,
         self.static_rotation,
         self.static_opacity,
         self.static_max_radii2D,
         self.static_xyz_gradient_accum
         # ==================================
        ) = model_args
    else:
        # 3D인 경우는 기존 방식 사용
        pass

    if training_args is not None:
        self.training_setup(training_args)
        self.xyz_gradient_accum = self.xyz_gradient_accum
        self.t_gradient_accum = self.t_gradient_accum
        self.denom = self.denom
        self.optimizer.load_state_dict(opt_dict)


def separate_static_gaussians(self, tau):
    """
    4D Gaussian 중 _scaling_t의 exp() 값이 tau를 초과하면 해당 Gaussian을 정적으로 전환합니다.
    전환 시, 시간 성분(_t)을 0으로 설정하고, 4D 회전 행렬에서 공간 부분만 추출하며, 
    시간 관련 파라미터(_scaling_t, 시간 SH 등)는 고정합니다.
    """
    if self.gaussian_dim != 4:
        print("모델이 3D이므로 정적/동적 분리는 적용되지 않습니다.")
        return

    temporal_scales = torch.exp(self._scaling_t).squeeze(-1)  # (N,)
    static_mask = temporal_scales > tau
    dynamic_mask = ~static_mask

    num_static = static_mask.sum().item()
    num_dynamic = dynamic_mask.sum().item()
    print(f"정적 Gaussian 개수: {num_static}, 동적 Gaussian 개수: {num_dynamic}")
    if num_static == 0:
        return

    # 정적 Gaussian 파라미터를 추가
    self.static_xyz = torch.cat([self.static_xyz, self._xyz[static_mask]], dim=0)
    self.static_features_dc = torch.cat([self.static_features_dc, self._features_dc[static_mask]], dim=0)
    self.static_features_rest = torch.cat([self.static_features_rest, self._features_rest[static_mask]], dim=0)
    self.static_scaling = torch.cat([self.static_scaling, self._scaling[static_mask]], dim=0)
    self.static_rotation = torch.cat([self.static_rotation, self._rotation[static_mask]], dim=0)
    self.static_opacity = torch.cat([self.static_opacity, self._opacity[static_mask]], dim=0)
    self.static_max_radii2D = torch.cat([self.static_max_radii2D, self.max_radii2D[static_mask]], dim=0)
    self.static_xyz_gradient_accum = torch.cat([self.static_xyz_gradient_accum, self.xyz_gradient_accum[static_mask]], dim=0)

    # 동적 Gaussian 집합에서 정적 항목 제거 (시간 관련 파라미터도 함께 제거)
    self._xyz = self._xyz[dynamic_mask]
    self._features_dc = self._features_dc[dynamic_mask]
    self._features_rest = self._features_rest[dynamic_mask]
    self._scaling = self._scaling[dynamic_mask]
    self._rotation = self._rotation[dynamic_mask]
    self._opacity = self._opacity[dynamic_mask]
    self.max_radii2D = self.max_radii2D[dynamic_mask]
    self.xyz_gradient_accum = self.xyz_gradient_accum[dynamic_mask]
    self._t = self._t[dynamic_mask]
    self._scaling_t = self._scaling_t[dynamic_mask]
    if self.rot_4d:
        self._rotation_r = self._rotation_r[dynamic_mask]

def get_combined_means(self, timestamp):
    """
    주어진 timestamp에서 동적 Gaussian은 시간 offset을 적용하여 3D 좌표를 계산하고,
    정적 Gaussian은 이미 3D 상태이므로 그대로 반환한 후, 두 그룹을 합쳐 반환한다.
    """
    dynamic_means = self._xyz.clone()
    if self.gaussian_dim == 4:
        _, delta_mean = self.get_current_covariance_and_mean_offset(1.0, timestamp)
        if hasattr(self, 'static_mask') and self.static_mask.any():
            delta_mean[self.static_mask] = 0.0
        dynamic_means += delta_mean
    # 정적 Gaussian의 좌표는 self.static_xyz에 저장되어 있음
    static_means = self.static_xyz
    return torch.cat([dynamic_means, static_means], dim=0)

def get_combined_covariance(self, scaling_modifier=1.0, timestamp=None):
    """
    주어진 timestamp에서 동적 Gaussian과 정적 Gaussian의 3×3 공분산을 결합하여 반환한다.
    """
    if self.gaussian_dim == 4:
        cov, _ = self.get_current_covariance_and_mean_offset(scaling_modifier, timestamp)
        if hasattr(self, 'static_mask') and self.static_mask.any():
            static_idx = self.static_mask
            cov_static = GaussianModel.build_covariance_from_scaling_rotation(
                scaling_modifier * self._scaling[static_idx],
                torch.eye(3, device=cov.device).reshape(1, 3, 3).repeat(static_idx.sum(), 1, 1)
            )
            cov[static_idx] = cov_static
        dynamic_cov = cov
    else:
        dynamic_cov = self.get_covariance(scaling_modifier)
    
    # 정적 Gaussian의 공분산 계산 (이미 3D이므로)
    static_cov = GaussianModel.build_covariance_from_scaling_rotation(
        self.static_scaling.exp(),  # scaling_activation 적용된 값
        self.static_rotation
    )
    return torch.cat([dynamic_cov, static_cov], dim=0)


# # ???
# def get_all_opacities(self):
#     dynamic_opacity = self.opacity_activation(self._opacity)
#     static_opacity = self.opacity_activation(self.static_opacity)
#     return torch.cat([dynamic_opacity, static_opacity], dim=0)

# def get_all_sh_features(self):
#     dynamic_sh = torch.cat((self._features_dc, self._features_rest), dim=2)
#     static_sh = torch.cat((self.static_features_dc, self.static_features_rest), dim=2)
#     return torch.cat([dynamic_sh, static_sh], dim=0)

