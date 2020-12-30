import os
import torch
import datetime

class ModelHolder():
    """
    `ModelHolder` is a model manager.\n
    You can manual save model by using `self.save_checkpoint`,\n
    and `ModelHolder` also auto save model while the end of `ModelHolder`.\n
    All save model will store under `./.model_holder`.\n
    Usage Example:
        `
        with ModelHolder(model) as (model_holder,model):            
        `
    """

    def __init__(self,model):
        """
        Args:
            model (class<torch.nn.Module>): model
        """
        self.model = model
    
    def save_checkpoint(self,save_name = None):
        """
        save checkpoint manual
        Args:
            save_name (str): name to save, model will store under .model_holder
        """
        if not os.path.isdir('.model_holder'):
            os.mkdir('.model_holder')

        if save_name is None:
            save_name = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S.bin")
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        model_to_save.config.save_pretrained(".model_holder/")
        torch.save(model_to_save, '.model_holder/%s'%save_name)
    
    def __enter__(self):
        """
        Returns:
            self,self.model
        """
        return self,self.model
    
    def __exit__(self,exc_type, exc_val, exc_tb):
        """
        auto save checkpoint while exit ModelHolder
        """

        if exc_type is KeyboardInterrupt:
            print("Catch KeyboardInterrupt, model will auto save in .model_holder")
        
        self.save_checkpoint()
    
    def save_checkpoint2(self,model,save_path,save_name=None):
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        if save_name is None:
            save_name = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S.bin")
        # model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.config.save_pretrained(save_path)
        torch.save(model_to_save, save_path+save_name)
