def get_combination(folder, combination_name, save_name):
    combinations = []
    with open(folder, 'r') as f:
        file_content_list = f.readlines()
        for combination in combination_name:
            for row in file_content_list:
                row_list = row.split(',')
                if combination in (row_list[0]+',' + row_list[1]):
                    combinations.append(row)

    with open(save_name, 'w') as f:
        for row in combinations:
            f.write(row[:12] + row[14:])

if __name__ == "__main__":
    folder = 'MLR_combination.txt'

    combination_name = [
        'BC+N+FS+FR+HT,Alzheimers Handwriting',
        'BC+N+FR+HT,Alzheimers Handwriting',
        'BC+N+FR+HT,Breast Cancer',
        'BC+N+DR+FR+HT+Bag offset,Spam Email',
        'BC+N+PolyK+FR+HT,Water Potability',
        'BC+N+PolyK+FR+HT+Bag offset,Water Potability',
    ]

    get_combination(folder, combination_name, 'MLR_combination_selection.txt')
