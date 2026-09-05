from models.data_quality import validate_applicant_values


def test_data_quality_warning():
    warnings = validate_applicant_values({"annual_inc": 0, "dti": 120, "revol_util": 200})
    assert len(warnings) == 3
