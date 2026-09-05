from __future__ import annotations

from typing import Any, Mapping


def validate_applicant_values(applicant: Mapping[str, Any]) -> list[str]:
    """Return non-fatal data-quality warnings for analyst visibility."""
    warnings: list[str] = []
    non_negative = ["loan_amnt", "installment", "annual_inc", "revol_bal", "total_acc", "open_acc", "delinq_2yrs", "inq_last_6mths", "pub_rec"]
    for field in non_negative:
        if field in applicant and float(applicant[field]) < 0:
            warnings.append(f"{field} is negative")
    if "dti" in applicant and not 0 <= float(applicant["dti"]) <= 100:
        warnings.append("dti is outside the expected 0–100 range")
    if "revol_util" in applicant and not 0 <= float(applicant["revol_util"]) <= 150:
        warnings.append("revol_util is outside the expected 0–150 range")
    if "annual_inc" in applicant and float(applicant["annual_inc"]) == 0:
        warnings.append("annual_inc is zero; interpret the prediction cautiously")
    return warnings
