import re

def cost_column_to_float(df):
    def make_cost_float(row):
        cost = row['project_cost']
        cost = re.sub(r'[$,]', '', cost)
        return float(cost)
    
    df['project_cost'] = df.apply(make_cost_float, axis=1)
    return df