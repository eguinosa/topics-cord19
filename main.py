# Gelin Eguinosa Rosique

from topic_model import TopicModel
from time_keeper import TimeKeeper
from extra_funcs import big_number


# Record Runtime of the Program.
stopwatch = TimeKeeper()

# Load the available Topic Models.
saved_models = TopicModel.saved_topic_models()
print("\nAvailable Topic Models:")
for model_id in saved_models:
    print(f"\nTopic Model <{model_id}>:")
    basic_info = TopicModel.basic_info(model_id)
    # Display Basic Info of the Model.
    print("  Topics:", big_number(basic_info['num_topics']))
    print("  Document Model:", basic_info['model_type'].title())
    print("  Corpus Size:", big_number(basic_info['corpus_size']))
    hierarchy = 'Yes' if basic_info['topics_hierarchy'] else 'No'
    print("  Topic Hierarchy:", hierarchy)

print("\nDone.")
print(f"[{stopwatch.formatted_runtime()}]\n")

# Load One of the Topic Models.

# Load the Vocabulary of the Model.

print("\nDone.")
print(f"[{stopwatch.formatted_runtime()}]\n")
