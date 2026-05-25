import pandas as pd

def prepare_input(transaction_type, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest):
    """
    Formats the user input into a DataFrame that the model expects.
    """
    return pd.DataFrame([{
        "type": transaction_type,
        "amount": amount,
        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrig,
        "oldbalanceDest": oldbalanceDest,
        "newbalanceDest": newbalanceDest,
    }])