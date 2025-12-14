from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class GenderEnum(str, Enum):
    female = "female"
    male = "male"


class FamilyHistoryEnum(str, Enum):
    yes = "yes"
    no = "no"


class FavcEnum(str, Enum):
    yes = "yes"
    no = "no"


class CaecEnum(str, Enum):
    no = "no"
    sometimes = "sometimes"
    frequently = "frequently"
    always = "always"


class SmokeEnum(str, Enum):
    yes = "yes"
    no = "no"


class SccEnum(str, Enum):
    yes = "yes"
    no = "no"


class CalcEnum(str, Enum):
    no = "no"
    sometimes = "sometimes"
    frequently = "frequently"
    always = "always"


class MtransEnum(str, Enum):
    automobile = "automobile"
    motorbike = "motorbike"
    bike = "bike"
    public_transportation = "public_transportation"
    walking = "walking"


class PatientData(BaseModel):
    gender: GenderEnum = Field(..., description="Gênero do paciente")
    age: float = Field(..., gt=0, le=120, description="Idade em anos")
    # height: float = Field(..., gt=0.5, lt=3.0, description="Altura em metros")
    # weight: float = Field(..., gt=10.0, lt=300.0, description="Peso em kg")
    family_history: FamilyHistoryEnum = Field(
        ..., description="Histórico familiar de obesidade"
    )
    favc: FavcEnum = Field(..., description="Consumo frequente de alimentos calóricos")
    fcvc: float = Field(..., ge=1, le=3, description="Consumo de vegetais (1-3)")
    ncp: float = Field(..., ge=1, le=4, description="Número de refeições principais")
    caec: CaecEnum = Field(..., description="Consumo de alimentos entre refeições")
    smoke: SmokeEnum = Field(..., description="Fumante")
    ch2o: float = Field(..., ge=1, le=3, description="Consumo diário de água (1-3)")
    scc: SccEnum = Field(..., description="Monitoramento de calorias")
    faf: float = Field(
        ...,
        ge=0,
        le=3,
        description="Frequência de atividade física (0-3)",
    )
    tue: float = Field(
        ...,
        ge=0,
        le=2,
        description="Tempo usando dispositivos tecnológicos (0-2)",
    )
    calc: CalcEnum = Field(..., description="Consumo de álcool")
    mtrans: MtransEnum = Field(..., description="Meio de transporte principal")

    model_config = ConfigDict(
        use_enum_values=True,
        json_schema_extra={
            "example": {
                "gender": "male",
                "age": 25,
                # "height": 1.75,
                # "weight": 70.0,
                "family_history": "yes",
                "favc": "yes",
                "fcvc": 2.0,
                "ncp": 3.0,
                "caec": "sometimes",
                "smoke": "no",
                "ch2o": 2.0,
                "scc": "no",
                "faf": 1.0,
                "tue": 0.0,
                "calc": "sometimes",
                "mtrans": "public_transportation",
            }
        },
    )


class PredictionResponse(BaseModel):
    label: str = Field(..., description="Classificação de obesidade prevista")
    probability: float = Field(..., description="Confiança do modelo na predição")
    model_version: str = Field(..., description="Versão do modelo utilizado")
