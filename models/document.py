# models 数据库模型
from tortoise import fields, models
from pydantic import BaseModel
from datetime import datetime

class Document(models.Model):
    id = fields.IntegerField(pk=True)
    file_uuid = fields.CharField(max_length=64, unique=True)
    file_name = fields.CharField(max_length=255)
    save_path = fields.CharField(max_length=512)
    file_size = fields.IntField(null=True)
    created_at = fields.DatetimeField(auto_now_add=True)

    class Meta:
        table = "document"

# Pydantic返回模型
class DocumentResp(BaseModel):
    id: int
    file_uuid: str
    file_name: str
    save_path: str
    file_size: int | None
    created_at: datetime

    class Config:
        from_attributes = True
