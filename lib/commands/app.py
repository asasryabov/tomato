# -*- coding: utf-8 -*-
import sys
from lib.trainer import Trainer

class App(object):
    """Tomato AI application"""
    
    def train(self,
        data_set="./set",
        validation_set="./validation_set",
        image_size=50,
        epochs=100,
        verbose=0 ):
        """train AI with training set"""
        trainer = Trainer(
            data_set=data_set,
            validation_set=validation_set,
            image_size=image_size,
        )
        trainer.train(
            epochs=epochs,
            verbose=verbose
        )
