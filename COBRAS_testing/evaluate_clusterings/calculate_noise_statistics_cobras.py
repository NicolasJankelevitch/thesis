import itertools
from pathlib import Path
import jsonpickle

def calculate_noise_statistics(result_path, max_index = None, max_number_of_results = None):
    path = result_path
    total_number_of_mistakes = 0
    total_number_of_mistakes_in_last_result = 0
    total_number_of_last_results = 0
    total_number_of_intermediate_results = 0
    if max_number_of_results is not None:
        generator = itertools.islice(path.iterdir(),max_number_of_results)
    else:
        generator = path.iterdir()
    for file in generator:
        try:
            with file.open(mode='r') as input_file:
                string = input_file.readline()
                result_dir =jsonpickle.decode(string)
                corrected_constraint_set = result_dir["corrected_constraint_sets"]

            for (index, relevant, mistakes) in corrected_constraint_set:
                if max_index is not None and index>max_index:
                    break
                total_number_of_intermediate_results+=1
                if len(mistakes) > 0:
                    total_number_of_mistakes += 1

            total_number_of_last_results+=1
            if len(corrected_constraint_set[-1][2])>0:
                total_number_of_mistakes_in_last_result+=1
        except Exception as e:
            pass
            # print("exception in ", result_directory_name.name, " file ", file.name)
    p_any_mistake = total_number_of_mistakes/total_number_of_intermediate_results
    p_last_mistake = total_number_of_mistakes_in_last_result/total_number_of_last_results
    return p_any_mistake, p_last_mistake



