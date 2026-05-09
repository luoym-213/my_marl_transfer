class ModuleAccessors:
    @property
    def low_agent_encoder(self):
        return self.modules_dict['low_level']['encoder']
    
    @property
    def value_head(self):
        return self.modules_dict['low_level']['value_head']
    
    @property
    def policy_head(self):
        return self.modules_dict['low_level']['policy_head']
    
    @property
    def dist(self):
        return self.modules_dict['low_level']['dist']
    
    @property
    def map_conv1(self):
        return self.modules_dict['high_level']['map_conv1']
    
    @property
    def map_conv2(self):
        return self.modules_dict['high_level']['map_conv2']
    
    @property
    def map_conv3(self):
        return self.modules_dict['high_level']['map_conv3']
    
    @property
    def vec_mlp(self):
        return self.modules_dict['high_level']['vec_mlp']
    
    @property
    def decision_head(self):
        return self.modules_dict['high_level']['decision_head']
    
    @property
    def decoder_fuse(self):
        return self.modules_dict['high_level']['decoder_fuse']
    
    @property
    def decoder_up1(self):
        return self.modules_dict['high_level']['decoder_up1']
    
    @property
    def decoder_up2(self):
        return self.modules_dict['high_level']['decoder_up2']
    
    @property
    def decoder_out(self):
        return self.modules_dict['high_level']['decoder_out']
    
    @property
    def critic_map_backbone(self):
        return self.modules_dict['high_critic']['map_backbone']
    
    @property
    def critic_map_compress(self):
        return self.modules_dict['high_critic']['map_compress']
    
    @property
    def critic_agent_encoder(self):
        return self.modules_dict['high_critic']['agent_encoder']
    
    @property
    def critic_fusion_layer(self):
        return self.modules_dict['high_critic']['fusion_layer']
    
    @property
    def critic_value_out_heads(self):
        return self.modules_dict['high_critic']['value_heads']
    
    @property
    def critic_map_flat_dim(self):
        return 64 * 6 * 6
    
    @property
    def ego_node_encoder(self):
        return self.modules_dict['high_level']['ego_node_encoder']
    
    @property
    def teammate_node_encoder(self):
        return self.modules_dict['high_level']['teammate_node_encoder']
    
    @property
    def explore_node_encoder(self):
        return self.modules_dict['high_level']['explore_node_encoder']
    
    @property
    def landmark_node_encoder(self):
        return self.modules_dict['high_level']['landmark_node_encoder']
    
    @property
    def explore_node_linear(self):
        return self.modules_dict['high_level']['explore_node_linear']
    
    @property
    def landmark_node_linear(self):
        return self.modules_dict['high_level']['landmark_node_linear']
    
    @property
    def linear_ln(self):
        return self.modules_dict['high_level']['linear_ln']
    
    @property
    def edge_encoder(self):
        return self.modules_dict['high_level']['edge_encoder']
    
    @property
    def q_proj(self):
        return self.modules_dict['high_level']['q_proj']
    
    @property
    def k_proj(self):
        return self.modules_dict['high_level']['k_proj']
    
    @property
    def v_proj(self):
        return self.modules_dict['high_level']['v_proj']
    
    @property
    def node_selection_head(self):
        return self.modules_dict['high_level']['node_selection_head']
    
    @property
    def attn_dim(self):
        return 64
