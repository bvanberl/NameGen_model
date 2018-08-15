import csv


def create_files():
    male_names = []
    female_names = []

    # Read in all the names in the data set
    for year in range (1880, 2017):
        with open('input/names/yob' + str(year) + '.txt') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                name = row[0].lower()
                if row[1] == 'M':
                    male_names.append(name)
                else:
                    female_names.append(name)

    # Convert names to a set to preserve only unique entries
    male_names = set(male_names)
    female_names = set(female_names)

    # Output the names to text files, separated by \n
    with open('input/male_names.txt', mode='wt', encoding='utf-8') as male_file:
        male_file.write('\n'.join(male_names))
    with open('input/female_names.txt', mode='wt', encoding='utf-8') as female_file:
        female_file.write('\n'.join(female_names))

    print("We have a list of " + str(len(male_names)) + " male names and " + str(len(female_names)) + " female names.")


create_files()