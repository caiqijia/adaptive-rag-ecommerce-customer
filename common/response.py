from pydantic import BaseModel
from typing import Any,Optional

class Resp(BaseModel):
    code : int
    msg : str
    data : Optional[Any] = None

def success_resp(data:Any=None,msg:str="success") -> Resp:
    return Resp(code=200,msg=msg,data=data
    )

def fail_resp(code : int = 500,msg : str = "fail",data : Any =None) -> Resp:
    return Resp(
        code=code,msg=msg,data=data
    )

