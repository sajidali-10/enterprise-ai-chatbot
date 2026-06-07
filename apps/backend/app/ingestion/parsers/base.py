from abc import ABC, abstractmethod

class BaseParser(ABC):
    @abstractmethod
    def parse(self, file_bytes: bytes) -> str: ...
    
    @property
    @abstractmethod
    def supported_types(self) -> list[str]: ...