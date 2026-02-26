"""统一存储入口"""

from .base import StorageProvider, UploadResult
from .r2 import R2Storage

__all__ = ["StorageProvider", "UploadResult", "R2Storage"]
