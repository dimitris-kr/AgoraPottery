from pydantic import BaseModel

class ModelSchema(BaseModel):
    id: int
    name: str

    class Config:
        from_attributes = True

class ModelVersionSchema(BaseModel):
    id: int
    model: ModelSchema
    version: str
    train_sample_size: int | None
    val_loss: float | None
    val_accuracy: float | None
    val_mae: float | None

    class Config:
        from_attributes = True