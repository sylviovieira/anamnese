
def categorize_age(age):
    try:
        age = int(age)
        if age < 30:
            return "Young"
        elif age < 45:
            return "Middle"
        elif age < 60:
            return "Old"
        else:
            return "Elderly"
    except:
        return "Unknown"


def categorize_bmi(bmi):
    try:
        bmi = float(bmi)
        if bmi < 18.5:
            return "Low"
        elif bmi < 25:
            return "Normal"
        elif bmi < 30:
            return "High"
        else:
            return "Very High"
    except:
        return "Unknown"
