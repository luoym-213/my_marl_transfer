import torch


class Checkpointing:
    def get_module_params(self, module_name):
        """
        获取指定模块的参数
        
        Args:
            module_name: 'shared', 'low_level', 'high_level', 'high_critic'
        
        Returns:
            list of parameters
        """
        if module_name not in self.modules_dict:
            raise ValueError(f"Module '{module_name}' not found! Available: {list(self.modules_dict.keys())}")
        
        return list(self.modules_dict[module_name].parameters())

    def freeze_module(self, module_name):
        """冻结指定模块的参数"""
        for param in self.get_module_params(module_name):
            param.requires_grad = False
        print(f"✅ Module '{module_name}' frozen")

    def unfreeze_module(self, module_name):
        """解冻指定模块的参数"""
        for param in self.get_module_params(module_name):
            param.requires_grad = True
        print(f"✅ Module '{module_name}' unfrozen")

    def save_module_checkpoint(self, module_name, path):
        """
        保存指定模块的参数
        
        Args:
            module_name: 模块名称
            path: 保存路径
        """
        if module_name not in self.modules_dict:
            raise ValueError(f"Module '{module_name}' not found!")
        
        checkpoint = {
            'module_name': module_name,
            'state_dict': self.modules_dict[module_name].state_dict(),
            'config': {
                'num_agents': self.num_agents,
                'num_entities': self.num_entities,
                'hidden_dim': self.h_dim,
                'embed_dim': self.embed_dim
            }
        }
        
        torch.save(checkpoint, path)
        print(f"✅ Saved '{module_name}' checkpoint to {path}")

    def load_module_checkpoint(self, module_name, path, strict=True, freeze=False):
        """
        加载指定模块的参数
        
        Args:
            module_name: 模块名称
            path: checkpoint 路径
            strict: 是否严格匹配参数
            freeze: 是否加载后冻结
        
        Returns:
            missing_keys, unexpected_keys
        """
        checkpoint = torch.load(path, map_location='cpu')
        
        # 验证配置
        if 'config' in checkpoint:
            config = checkpoint['config']
            if config.get('num_agents') != self.num_agents:
                print(f"⚠️ Warning: num_agents mismatch! "
                      f"Checkpoint: {config['num_agents']}, Current: {self.num_agents}")
        
        # 加载参数
        missing, unexpected = self.modules_dict[module_name].load_state_dict(
            checkpoint['state_dict'], 
            strict=strict
        )
        
        if freeze:
            self.freeze_module(module_name)
        
        print(f"✅ Loaded '{module_name}' checkpoint from {path}")
        if missing:
            print(f"  Missing keys: {missing}")
        if unexpected:
            print(f"  Unexpected keys: {unexpected}")
        
        return missing, unexpected

    def save_all_modules(self, save_dir):
        """保存所有模块到指定目录"""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        for module_name in self.modules_dict.keys():
            save_path = os.path.join(save_dir, f"{module_name}.pth")
            self.save_module_checkpoint(module_name, save_path)

    def load_pretrained_low_level(self, path, freeze=True):
        """
        智能加载函数：支持加载 '模块化Checkpoint' 或 '完整训练Checkpoint'
        """
        print(f"🔄 Loading low-level params from {path}...")
        checkpoint = torch.load(path, map_location='cpu')
        
        low_level_state_dict = {}
        
        # === 情况 A: 这是一个模块化 Checkpoint ===
        if 'state_dict' in checkpoint:
            print("  Type: Module Checkpoint")
            low_level_state_dict = checkpoint['state_dict']
            
        # === 情况 B: 这是一个完整训练 Checkpoint ===
        elif 'models' in checkpoint:
            print("  Type: Full Training Checkpoint (extracting params...)")
            full_state_dict = checkpoint['models'][0]
            
            # ⭐ 关键修改：明确底层网络的键名前缀
            # 旧代码中，底层网络的键名应该是 'low_agent_encoder.*', 'value_head.*' 等
            target_keys = [
                'low_agent_encoder',  # ← 这是底层编码器的真正名字
                'value_head',
                'policy_head',
                'dist'
            ]
            
            for key, value in full_state_dict.items():
                # 去除可能的 'modules_dict.low_level.' 前缀（如果是新版代码保存的）
                clean_key = key.replace('modules_dict.low_level.', '')
                
                # 检查是否属于底层网络（必须完整匹配前缀）
                if any(clean_key.startswith(prefix) for prefix in target_keys):
                    # ⭐ 如果是旧代码，需要将 'low_agent_encoder' 映射为 'encoder'
                    # 因为新代码中底层模块内部的名字是 'encoder'
                    final_key = clean_key.replace('low_agent_encoder', 'encoder')
                    low_level_state_dict[final_key] = value
                    
        else:
            raise ValueError(f"Unknown checkpoint format! Keys found: {list(checkpoint.keys())}")

        # 加载参数
        missing, unexpected = self.modules_dict['low_level'].load_state_dict(
            low_level_state_dict, 
            strict=False 
        )
        
        if freeze:
            self.freeze_module('low_level')
            
        print(f"✅ Low-level loaded. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
        if missing:
            print(f"  ⚠️ Missing: {missing}")
        if unexpected:
            print(f"  ⚠️ Unexpected: {unexpected}")
        
        return missing, unexpected
        """
        便捷函数：加载预训练的底层网络
        
        Args:
            path: checkpoint 路径
            freeze: 是否冻结参数
        """
        # return self.load_module_checkpoint('low_level', path, strict=False, freeze=freeze)

    def load_pretrained_high_level(self, path, freeze=False):
        """
        加载预训练的高层策略（Actor + Critic）
        """
        print(f"🔄 Loading high-level params from {path}...")
        checkpoint = torch.load(path, map_location='cpu')
        
        high_level_state_dict = {}
        high_critic_state_dict = {}
        
        # === 情况 A: 模块化 Checkpoint ===
        if 'state_dict' in checkpoint:
            module_name = checkpoint.get('module_name', None)
            if module_name == 'high_level':
                high_level_state_dict = checkpoint['state_dict']
            elif module_name == 'high_critic':
                high_critic_state_dict = checkpoint['state_dict']

        # === 情况 B: 完整 Checkpoint ===
        elif 'models' in checkpoint:
            print("  Type: Full Training Checkpoint (extracting high-level params...)")
            full_state_dict = checkpoint['models'][0]
            
            for key, value in full_state_dict.items():
                if 'modules_dict.high_level.' in key:
                    new_key = key.replace('modules_dict.high_level.', '')
                    high_level_state_dict[new_key] = value
                elif 'modules_dict.high_critic.' in key:
                    new_key = key.replace('modules_dict.high_critic.', '')
                    high_critic_state_dict[new_key] = value
        
        # 执行加载
        if high_level_state_dict:
            m, u = self.modules_dict['high_level'].load_state_dict(high_level_state_dict, strict=False)
            print(f"  ✅ High-Level Actor loaded. Missing: {len(m)}, Unexpected: {len(u)}")
            if freeze: self.freeze_module('high_level')
            
        if high_critic_state_dict:
            m, u = self.modules_dict['high_critic'].load_state_dict(high_critic_state_dict, strict=False)
            print(f"  ✅ High-Level Critic loaded. Missing: {len(m)}, Unexpected: {len(u)}")
            if freeze: self.freeze_module('high_critic')

    def get_trainable_params_by_modules(self, module_names, learning_rates=None):
        """
        获取多个模块的参数组（用于优化器）
        
        Args:
            module_names: 模块名称列表
            learning_rates: 对应的学习率列表（可选）
        
        Returns:
            param_groups for optimizer
        
        Example:
            >>> param_groups = model.get_trainable_params_by_modules(
            ...     ['high_level', 'high_critic', 'low_level'],
            ...     [3e-4, 3e-4, 1e-5]  # 底层使用更小的学习率
            ... )
            >>> optimizer = torch.optim.Adam(param_groups)
        """
        param_groups = []
        
        if learning_rates is None:
            learning_rates = [None] * len(module_names)
        
        for module_name, lr in zip(module_names, learning_rates):
            params = self.get_module_params(module_name)
            if lr is not None:
                param_groups.append({'params': params, 'lr': lr})
            else:
                param_groups.append({'params': params})
        
        return param_groups
