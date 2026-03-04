"""Patient model representing a patient's clinical data for diabetes prediction."""


class Patient:
    """Represents a patient with clinical attributes used for diabetes prediction.

    Attributes
    ----------
    pregnancies : int
        Number of times pregnant.
    glucose : float
        Plasma glucose concentration (mg/dL) from a 2-hour oral glucose tolerance test.
    blood_pressure : float
        Diastolic blood pressure (mm Hg).
    skin_thickness : float
        Triceps skinfold thickness (mm).
    insulin : float
        2-Hour serum insulin (mu U/ml).
    bmi : float
        Body mass index (kg/m²).
    diabetes_pedigree_function : float
        Diabetes pedigree function (a measure of genetic influence).
    age : int
        Age in years.
    """

    # Valid ranges based on the Pima Indians Diabetes Dataset
    _VALID_RANGES = {
        "pregnancies": (0, 20),
        "glucose": (0, 300),
        "blood_pressure": (0, 200),
        "skin_thickness": (0, 100),
        "insulin": (0, 900),
        "bmi": (0.0, 70.0),
        "diabetes_pedigree_function": (0.0, 3.0),
        "age": (0, 120),
    }

    def __init__(
        self,
        pregnancies: int,
        glucose: float,
        blood_pressure: float,
        skin_thickness: float,
        insulin: float,
        bmi: float,
        diabetes_pedigree_function: float,
        age: int,
    ) -> None:
        self.pregnancies = pregnancies
        self.glucose = glucose
        self.blood_pressure = blood_pressure
        self.skin_thickness = skin_thickness
        self.insulin = insulin
        self.bmi = bmi
        self.diabetes_pedigree_function = diabetes_pedigree_function
        self.age = age

    # ------------------------------------------------------------------
    # Properties with validation
    # ------------------------------------------------------------------

    @property
    def pregnancies(self) -> int:
        return self._pregnancies

    @pregnancies.setter
    def pregnancies(self, value: int) -> None:
        self._pregnancies = self._validate("pregnancies", int(value))

    @property
    def glucose(self) -> float:
        return self._glucose

    @glucose.setter
    def glucose(self, value: float) -> None:
        self._glucose = self._validate("glucose", float(value))

    @property
    def blood_pressure(self) -> float:
        return self._blood_pressure

    @blood_pressure.setter
    def blood_pressure(self, value: float) -> None:
        self._blood_pressure = self._validate("blood_pressure", float(value))

    @property
    def skin_thickness(self) -> float:
        return self._skin_thickness

    @skin_thickness.setter
    def skin_thickness(self, value: float) -> None:
        self._skin_thickness = self._validate("skin_thickness", float(value))

    @property
    def insulin(self) -> float:
        return self._insulin

    @insulin.setter
    def insulin(self, value: float) -> None:
        self._insulin = self._validate("insulin", float(value))

    @property
    def bmi(self) -> float:
        return self._bmi

    @bmi.setter
    def bmi(self, value: float) -> None:
        self._bmi = self._validate("bmi", float(value))

    @property
    def diabetes_pedigree_function(self) -> float:
        return self._diabetes_pedigree_function

    @diabetes_pedigree_function.setter
    def diabetes_pedigree_function(self, value: float) -> None:
        self._diabetes_pedigree_function = self._validate(
            "diabetes_pedigree_function", float(value)
        )

    @property
    def age(self) -> int:
        return self._age

    @age.setter
    def age(self, value: int) -> None:
        self._age = self._validate("age", int(value))

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _validate(self, field: str, value):
        """Validate that *value* is within the allowed range for *field*."""
        lo, hi = self._VALID_RANGES[field]
        if not (lo <= value <= hi):
            raise ValueError(
                f"'{field}' value {value} is out of range [{lo}, {hi}]."
            )
        return value

    def to_list(self) -> list:
        """Return patient features as an ordered list suitable for model input."""
        return [
            self.pregnancies,
            self.glucose,
            self.blood_pressure,
            self.skin_thickness,
            self.insulin,
            self.bmi,
            self.diabetes_pedigree_function,
            self.age,
        ]

    def to_dict(self) -> dict:
        """Return patient features as a dictionary."""
        return {
            "pregnancies": self.pregnancies,
            "glucose": self.glucose,
            "blood_pressure": self.blood_pressure,
            "skin_thickness": self.skin_thickness,
            "insulin": self.insulin,
            "bmi": self.bmi,
            "diabetes_pedigree_function": self.diabetes_pedigree_function,
            "age": self.age,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Patient":
        """Create a :class:`Patient` instance from a dictionary.

        Parameters
        ----------
        data : dict
            Must contain all eight feature keys.
        """
        return cls(
            pregnancies=data["pregnancies"],
            glucose=data["glucose"],
            blood_pressure=data["blood_pressure"],
            skin_thickness=data["skin_thickness"],
            insulin=data["insulin"],
            bmi=data["bmi"],
            diabetes_pedigree_function=data["diabetes_pedigree_function"],
            age=data["age"],
        )

    def __repr__(self) -> str:
        return (
            f"Patient(pregnancies={self.pregnancies}, glucose={self.glucose}, "
            f"blood_pressure={self.blood_pressure}, skin_thickness={self.skin_thickness}, "
            f"insulin={self.insulin}, bmi={self.bmi}, "
            f"diabetes_pedigree_function={self.diabetes_pedigree_function}, "
            f"age={self.age})"
        )
