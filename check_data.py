import pandas as pd

df = pd.read_csv('sport_data.csv')

def run_dq_tests(df):
    print("--- Data Quality Report ---")
    
    # 1. Null Check
    null_count = df.isnull().sum().sum()
    null_pass = null_count == 0
    print(f"1. Completeness: {'PASS' if null_pass else 'FAIL'} ({null_count} missing values)")
    
    # 2. Range Check (assuming skills are 0-10)
    skills = df.columns[1:11]
    out_of_range = ((df[skills] < 0) | (df[skills] > 10)).any().any()
    range_pass = not out_of_range
    print(f"2. Validity: {'PASS' if range_pass else 'FAIL'} (Values outside 0-10 found)")
    
    # 3. Duplicate Check
    duplicate_count = df['SPORT'].duplicated().sum()
    dup_pass = duplicate_count == 0
    print(f"3. Uniqueness: {'PASS' if dup_pass else 'FAIL'} ({duplicate_count} duplicates found)")

    # Final Boolean: Only returns True if ALL three tests pass
    all_passed = null_pass and range_pass and dup_pass
    return all_passed

# Usage
run_dq_tests(df)
