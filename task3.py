import numpy as np


def is_total_supply_equal_to_total_demand(supply, demand):
    return np.sum(supply) == np.sum(demand)


def initial_sum_of_costs(assignments, cost_matrix):
    num_rows, num_cols = cost_matrix.shape
    total_cost = 0
    for i in range(num_rows):
        for j in range(num_cols):
            total_cost += assignments[i, j] * cost_matrix[i, j]
    return total_cost


def string_to_numpy_array(input_string):
    return np.array([int(item) for item in input_string.split(' ')])


def get_user_input():
    supply_input = input("Enter the supply vector: ")
    demand_input = input("Enter the demand vector: ")

    supply_vector = string_to_numpy_array(supply_input)
    num_rows = len(supply_vector)
    demand_vector = string_to_numpy_array(demand_input)
    num_cols = len(demand_vector)

    cost_matrix = np.zeros((num_rows, num_cols))

    print("Enter the cost matrix row by row:")
    for i in range(num_rows):
        row_input = input(f"Row {i + 1}: ")
        cost_matrix[i] = string_to_numpy_array(row_input)

    return supply_vector, demand_vector, cost_matrix


def update_supply_demand(assignments, supply, demand):
    updated_supply = supply - np.sum(assignments, axis=1)
    updated_demand = demand - np.sum(assignments, axis=0)
    return updated_supply, updated_demand


def north_west_corner_method(supply, demand):
    i, j = 0, 0
    assignments = np.zeros((len(supply), len(demand)))

    while i < len(supply) and j < len(demand):
        assignments[i, j] = min(supply[i], demand[j])
        supply[i] -= assignments[i, j]
        demand[j] -= assignments[i, j]

        if supply[i] == 0:
            i += 1
        elif demand[j] == 0:
            j += 1

    return assignments


def vogels_approximation_method(supply, demand, cost_matrix):
    num_rows, num_cols = cost_matrix.shape
    cost_table = np.copy(cost_matrix)
    supply_column, demand_row = num_cols, num_rows
    deleted_rows, deleted_cols = set(), set()
    assignments = np.zeros_like(cost_matrix)

    # Add diff column and row
    cost_table = np.append(cost_table, np.zeros((num_rows, 1)), axis=1)
    cost_table = np.append(cost_table, np.zeros((1, num_cols + 1)), axis=0)

    while len(deleted_rows) < num_rows and len(deleted_cols) < num_cols:
        # Update diff column and row
        for row in range(num_rows):
            if row not in deleted_rows:
                costs = cost_table[row, :num_cols]
                cost_table[row, num_cols] = minimum_difference(costs, deleted_cols)
        for col in range(num_cols):
            if col not in deleted_cols:
                costs = cost_table[:num_rows, col]
                cost_table[num_rows, col] = minimum_difference(costs, deleted_rows)

        # Choose the highest diff
        max_diff_row = max((r for r in range(num_rows) if r not in deleted_rows),
                           key=lambda x: cost_table[x, num_cols], default=-1)
        max_diff_col = max((c for c in range(num_cols) if c not in deleted_cols),
                           key=lambda x: cost_table[num_rows, x], default=-1)

        if max_diff_row == -1 or max_diff_col == -1:
            # Break the loop if there are no more valid rows or columns
            break

        if cost_table[max_diff_row, num_cols] >= cost_table[num_rows, max_diff_col]:
            i, j = max_diff_row, minimum_index_in_row(
                cost_table, max_diff_row, num_cols, deleted_cols)
        else:
            i, j = minimum_index_in_column(
                cost_table, max_diff_col, num_rows, deleted_rows), max_diff_col

        # Assign supply or demand
        supply_demand_min = min(supply[i], demand[j])
        assignments[i, j] = supply_demand_min
        supply[i] -= supply_demand_min
        demand[j] -= supply_demand_min

        # Update deleted rows and columns
        if supply[i] == 0 and i not in deleted_rows:
            deleted_rows.add(i)
        if demand[j] == 0 and j not in deleted_cols:
            deleted_cols.add(j)

    return assignments


def russells_approximation_method(supply, demand, cost_matrix):
    num_rows, num_cols = cost_matrix.shape
    assignments = np.zeros_like(cost_matrix, dtype=float)
    u = np.full(num_rows, -np.inf)  # Max value in rows
    v = np.full(num_cols, -np.inf)  # Max value in columns
    remaining_supply = supply.copy()
    remaining_demand = demand.copy()

    while remaining_supply.sum() > 0 and remaining_demand.sum() > 0:
        # Update max values in rows and columns
        for i in range(num_rows):
            if remaining_supply[i] > 0:
                u[i] = max(cost_matrix[i, :])
        for j in range(num_cols):
            if remaining_demand[j] > 0:
                v[j] = max(cost_matrix[:, j])

        # Calculate Russell's cost difference and find max position
        max_value = -np.inf
        max_pos = (-1, -1)
        for i in range(num_rows):
            for j in range(num_cols):
                if remaining_supply[i] > 0 and remaining_demand[j] > 0:
                    russell_value = u[i] + v[j] - cost_matrix[i, j]
                    if russell_value > max_value:
                        max_value = russell_value
                        max_pos = (i, j)

        # Allocate at max position
        i, j = max_pos
        allocation = min(remaining_supply[i], remaining_demand[j])
        assignments[i, j] = allocation
        remaining_supply[i] -= allocation
        remaining_demand[j] -= allocation

    return assignments


def minimum_difference(costs, omit):
    lowest, second_lowest = np.inf, np.inf
    for i, c in enumerate(costs):
        if i in omit:
            continue
        elif c < lowest:
            second_lowest, lowest = lowest, c
        elif c < second_lowest:
            second_lowest = c
    return lowest if second_lowest == np.inf else second_lowest - lowest


def minimum_index_in_row(cost_table, i, supply_column, deleted_cols):
    costs = cost_table[i][:supply_column]
    costs_left = np.delete(costs, list(deleted_cols))
    lowest_cost = np.min(costs_left)
    j = list(set(np.where(costs == lowest_cost)[0]) - deleted_cols)[0]
    return j


def minimum_index_in_column(cost_table, j, demand_row, deleted_rows):
    costs = cost_table[:, j][:demand_row]
    costs_left = np.delete(costs, list(deleted_rows))
    lowest_cost = np.min(costs_left)
    i = list(set(np.where(costs == lowest_cost)[0]) - deleted_rows)[0]
    return i


def main():
    supply_vector, demand_vector, cost_matrix = get_user_input()

    # Check if the problem is balanced
    if not is_total_supply_equal_to_total_demand(supply_vector, demand_vector):
        print("The problem is not balanced!")
    else:
        print("Input Parameter Table:")
        print("Supply:", supply_vector)
        print("Demand:", demand_vector)
        print("Cost Matrix:\n", cost_matrix)

        # Calculating solutions
        nw_solution = north_west_corner_method(supply_vector.copy(), demand_vector.copy())
        nw_result = initial_sum_of_costs(nw_solution, cost_matrix)

        vogels_solution = vogels_approximation_method(supply_vector.copy(), demand_vector.copy(), cost_matrix)
        vogels_result = initial_sum_of_costs(vogels_solution, cost_matrix)

        russells_solution = russells_approximation_method(supply_vector.copy(), demand_vector.copy(), cost_matrix)
        russells_result = initial_sum_of_costs(russells_solution, cost_matrix)

        print("North-West Corner Solution:\n", nw_solution)
        print(f'Total Cost = {nw_result}\n')

        print("Vogel's Approximation Solution:\n", vogels_solution)
        print(f'Total Cost = {vogels_result}\n')

        print("Russell's Approximation Solution:\n", russells_solution)
        print(f'Total Cost = {russells_result}\n')


# Run the main function
if __name__ == "__main__":
    main()
