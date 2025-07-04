import numpy as np


class Optimizer:
    """Base class for all optimizers"""
    
    def __init__(self, parameters):
        self.parameters = list(parameters)
    
    def step(self):
        """Perform a single optimization step"""
        raise NotImplementedError("Subclasses must implement step method")
    
    def zero_grad(self):
        """Zero gradients of all parameters"""
        for param in self.parameters:
            param.zero_grad()


class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer"""
    
    def __init__(self, parameters, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(parameters)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocity = [np.zeros_like(p.data) for p in self.parameters]
    
    def step(self):
        """Perform a single optimization step"""
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue
            
            grad = param.grad.copy()
            
            # Add weight decay
            if self.weight_decay > 0:
                grad += self.weight_decay * param.data
            
            # Apply momentum
            if self.momentum > 0:
                self.velocity[i] = self.momentum * self.velocity[i] + grad
                grad = self.velocity[i]
            
            # Update parameters
            param.data = param.data - self.lr * grad
    
    def __repr__(self):
        return f"SGD(lr={self.lr}, momentum={self.momentum}, weight_decay={self.weight_decay})"


class Adam(Optimizer):
    """Adam optimizer"""
    
    def __init__(self, parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        super().__init__(parameters)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        
        # Initialize momentum buffers
        self.m = [np.zeros_like(p.data) for p in self.parameters]
        self.v = [np.zeros_like(p.data) for p in self.parameters]
        self.t = 0
    
    def step(self):
        """Perform a single optimization step"""
        self.t += 1
        
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue
            
            grad = param.grad.copy()
            
            # Add weight decay
            if self.weight_decay > 0:
                grad += self.weight_decay * param.data
            
            # Update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            
            # Update biased second raw moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)
            
            # Compute bias-corrected first moment estimate
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            
            # Compute bias-corrected second raw moment estimate
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            # Update parameters
            param.data = param.data - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
    
    def __repr__(self):
        return f"Adam(lr={self.lr}, betas={self.beta1, self.beta2}, eps={self.eps}, weight_decay={self.weight_decay})"


class AdamW(Optimizer):
    """AdamW optimizer (Adam with decoupled weight decay)"""
    
    def __init__(self, parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        super().__init__(parameters)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        
        # Initialize momentum buffers
        self.m = [np.zeros_like(p.data) for p in self.parameters]
        self.v = [np.zeros_like(p.data) for p in self.parameters]
        self.t = 0
    
    def step(self):
        """Perform a single optimization step"""
        self.t += 1
        
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue
            
            grad = param.grad.copy()
            
            # Update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            
            # Update biased second raw moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)
            
            # Compute bias-corrected first moment estimate
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            
            # Compute bias-corrected second raw moment estimate
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            # Update parameters with decoupled weight decay
            param.data = param.data - self.lr * (m_hat / (np.sqrt(v_hat) + self.eps) + self.weight_decay * param.data)
    
    def __repr__(self):
        return f"AdamW(lr={self.lr}, betas={self.beta1, self.beta2}, eps={self.eps}, weight_decay={self.weight_decay})"


class RMSprop(Optimizer):
    """RMSprop optimizer"""
    
    def __init__(self, parameters, lr=0.01, alpha=0.99, eps=1e-8, weight_decay=0.0):
        super().__init__(parameters)
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.weight_decay = weight_decay
        
        # Initialize squared gradient buffers
        self.v = [np.zeros_like(p.data) for p in self.parameters]
    
    def step(self):
        """Perform a single optimization step"""
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue
            
            grad = param.grad.copy()
            
            # Add weight decay
            if self.weight_decay > 0:
                grad += self.weight_decay * param.data
            
            # Update squared gradient moving average
            self.v[i] = self.alpha * self.v[i] + (1 - self.alpha) * (grad ** 2)
            
            # Update parameters
            param.data = param.data - self.lr * grad / (np.sqrt(self.v[i]) + self.eps)
    
    def __repr__(self):
        return f"RMSprop(lr={self.lr}, alpha={self.alpha}, eps={self.eps}, weight_decay={self.weight_decay})" 