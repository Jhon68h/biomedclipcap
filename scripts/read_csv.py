import pandas as pd

def get_unique_values_pandas(filename, column_name):
    df = pd.read_csv(filename)
    unique_values_array = df[column_name].unique()
    return list(unique_values_array)

def extract_number(value):
    return int(value.replace("case", ""))

def main():
    filename = "/workspace/experiments_colono/experiments_colono/clipclap_train_labeled_negative.csv"
    column_name = "case"

    unique_values = get_unique_values_pandas(filename, column_name)

    # ordenar de mayor a menor según el número
    unique_values_sorted = sorted(unique_values, key=extract_number, reverse=True)

    print("Valores únicos ordenados:")
    for value in unique_values_sorted:
        print(value)

if __name__ == "__main__":
    main()