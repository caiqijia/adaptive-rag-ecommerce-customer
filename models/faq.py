from tortoise import fields,models
from pydantic import BaseModel


class FAQ(models.Model):
    """FAQ知识库表，Adaptive路由命中直接返回答案"""
    id = fields.IntField(pk=True)
    question = fields.CharField(max_length=1024, description="FAQ问题")
    answer = fields.TextField(description="FAQ标准答案")
    class Meta:
        table = "faq"

class FAQResp(BaseModel):
    id:int
    question:str
    answer:str
    class Config:
        from_attributes = True