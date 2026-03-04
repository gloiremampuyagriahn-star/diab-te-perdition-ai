"""Unit tests for the Patient class."""

import pytest
from models.patient import Patient


# ---------------------------------------------------------------------------
# Construction and defaults
# ---------------------------------------------------------------------------

class TestPatientConstruction:
    def test_valid_construction(self):
        p = Patient(
            pregnancies=2,
            glucose=120,
            blood_pressure=70,
            skin_thickness=25,
            insulin=80,
            bmi=28.5,
            diabetes_pedigree_function=0.5,
            age=35,
        )
        assert p.pregnancies == 2
        assert p.glucose == 120.0
        assert p.blood_pressure == 70.0
        assert p.skin_thickness == 25.0
        assert p.insulin == 80.0
        assert p.bmi == 28.5
        assert p.diabetes_pedigree_function == 0.5
        assert p.age == 35

    def test_boundary_values_zero(self):
        p = Patient(0, 0, 0, 0, 0, 0.0, 0.0, 0)
        assert p.pregnancies == 0
        assert p.glucose == 0.0

    def test_boundary_values_max(self):
        p = Patient(20, 300, 200, 100, 900, 70.0, 3.0, 120)
        assert p.pregnancies == 20
        assert p.glucose == 300.0


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------

class TestPatientValidation:
    def test_negative_pregnancies_raises(self):
        with pytest.raises(ValueError):
            Patient(-1, 120, 70, 25, 80, 28.5, 0.5, 35)

    def test_glucose_too_high_raises(self):
        with pytest.raises(ValueError):
            Patient(0, 301, 70, 25, 80, 28.5, 0.5, 35)

    def test_age_too_high_raises(self):
        with pytest.raises(ValueError):
            Patient(0, 120, 70, 25, 80, 28.5, 0.5, 121)

    def test_bmi_negative_raises(self):
        with pytest.raises(ValueError):
            Patient(0, 120, 70, 25, 80, -1.0, 0.5, 35)

    def test_diabetes_pedigree_out_of_range_raises(self):
        with pytest.raises(ValueError):
            Patient(0, 120, 70, 25, 80, 28.5, 3.1, 35)


# ---------------------------------------------------------------------------
# Serialisation / deserialisation
# ---------------------------------------------------------------------------

class TestPatientSerialisation:
    def _make_patient(self):
        return Patient(2, 120, 70, 25, 80, 28.5, 0.5, 35)

    def test_to_list_length(self):
        p = self._make_patient()
        assert len(p.to_list()) == 8

    def test_to_list_order(self):
        p = self._make_patient()
        lst = p.to_list()
        assert lst[0] == 2      # pregnancies first
        assert lst[-1] == 35    # age last

    def test_to_dict_keys(self):
        p = self._make_patient()
        keys = set(p.to_dict().keys())
        expected = {
            "pregnancies", "glucose", "blood_pressure", "skin_thickness",
            "insulin", "bmi", "diabetes_pedigree_function", "age",
        }
        assert keys == expected

    def test_from_dict_roundtrip(self):
        p = self._make_patient()
        p2 = Patient.from_dict(p.to_dict())
        assert p.to_list() == p2.to_list()

    def test_repr(self):
        p = self._make_patient()
        assert "Patient" in repr(p)
        assert "glucose=120" in repr(p)
