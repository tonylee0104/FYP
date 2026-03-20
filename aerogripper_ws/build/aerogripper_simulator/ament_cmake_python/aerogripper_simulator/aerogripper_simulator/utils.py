"""低通滤波器工具（用于加速度数值微分）"""
import numpy as np

class LowPassFilter:
    """一阶低通滤波器"""
    def __init__(self, tau, dt, initial_value=0.0):
        """
        Args:
            tau: 滤波时间常数 (s)
            dt: 采样时间 (s)
            initial_value: 初始输出值
        """
        self.alpha = dt / (dt + tau) if tau > 0 else 1.0
        self.y = initial_value
    
    def update(self, x):
        """更新滤波器: y = alpha*x + (1-alpha)*y_prev"""
        self.y = self.alpha * x + (1 - self.alpha) * self.y
        return self.y