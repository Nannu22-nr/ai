def min_changes_to_good_matrix(t, test_cases):
    results = []
    for case in test_cases:
        n, m, matrix = case
        row_counts = [0] * n
        col_counts = [0] * m
        
        for i in range(n):
            for j in range(m):
                if matrix[i][j] == '1':
                    row_counts[i] += 1
                    col_counts[j] += 1
        
        row_odd = sum(1 for count in row_counts if count % 2 != 0)
        col_odd = sum(1 for count in col_counts if count % 2 != 0)
        
        if row_odd % 2 != col_odd % 2:
            results.append(-1)
        else:
            changes = max(row_odd, col_odd)
            results.append(changes)
    
    return results

# Read input
t = int(input())
test_cases = []
for _ in range(t):
    n, m = map(int, input().split())
    matrix = [input().strip() for _ in range(n)]
    test_cases.append((n, m, matrix))

# Compute results
results = min_changes_to_good_matrix(t, test_cases)

# Output results
for result in results:
    print(result)