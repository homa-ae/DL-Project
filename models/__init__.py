def get_model_class(model_type):
    model_type = model_type.lower()
    module = __import__(f"models.{model_type}", fromlist=[""])
    return getattr(module, model_type.upper())