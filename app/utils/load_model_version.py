import logging

logger = logging.getLogger(__name__)


# BUilding Models
def build_small_model():
    from app.build.train.v1.build_model import model, train_loader, train_data, device, num_classes, test_loader
    logger.debug(f'Deep learning model(small) built sucessfully.')
    return model, train_loader, train_data, device, num_classes, test_loader


def build_medium_model():
    from app.build.train.v3.build_model import model, train_loader, train_data, device, num_classes, test_loader
    logger.debug(f'Deep learning model(medium) built sucessfully.')
    return model, train_loader, train_data, device, num_classes, test_loader


def build_large_model():
    from app.build.train.v2.build_model import model, train_loader, train_data, device, num_classes, test_loader
    logger.debug(f'Deep learning model(large) built sucessfully.')
    return model, train_loader, train_data, device, num_classes, test_loader

def build_model(model_type: str):
      build_small_model() if model_type == "small" else build_medium_model() if model_type == "medium" else build_large_model()

# Loading Models

def large_model():
        from app.build.train.v2.load_model import model
        logger.info("Initialized Large Model")
        return model
def medium_model():
        from app.build.train.v3.load_model import model 
        logger.info("Initialized Medium Model")
        return model
        
def small_model():
        from app.build.train.v1.load_model_v1 import model
        logger.info("Initialized Small Model")
        return model

MODEL_BUILDERS = {
      "small": small_model,
      "medium": medium_model,
      "large": large_model
}

def load_model(model_type: str):
    MODEL_BUILDERS[model_type]()


