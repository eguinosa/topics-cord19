# Gelin Eguinosa Rosique

from topic_model import TopicModel
from simple_topic_model import SimpleTopicModel
from time_keeper import TimeKeeper
from extra_funcs import big_number


# Record Runtime of the Program.
stopwatch = TimeKeeper()

# Load the available Topic Models.
saved_models = TopicModel.saved_topic_models()
# Sort Models alphabetically
saved_models.sort()

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
user_input = input("\nWhich Topic Model would you like to load?\n([q/quit] to exit) ")
while True:
    user_input = user_input.strip().lower()
    if user_input in {'', 'q', 'quit', 'exit'}:
        exit()
    elif user_input in saved_models:
        break
    else:
        user_input = input("Please, type the Model's name again:\n ")

# Load Requested Model.
print(f"\nLoading Topic model <{user_input}>...")
model = SimpleTopicModel(model_id=user_input, show_progress=True)
print("\nDone.")
print(f"[{stopwatch.formatted_runtime()}]\n")

# Show Topics.
num_topics = model.num_topics
print(f"\n{num_topics} topics loaded.")
# Topic's Sizes.
print("\nTopic's Sizes:")
topic_sizes = model.top_topics()
for topic_and_size in topic_sizes:
    print(topic_and_size)

# Topic's Vocabulary.
user_input = input("\nShow vocabulary? (yes/[no]) ")
user_input = user_input.strip().lower()
show_vocab = True if user_input in {'y', 'yes'} else False
if show_vocab:
    top_n = 15
    print(f"\nTop {top_n} words per topic:")
    words_per_topic = model.all_topics_top_words(top_n)
    for i, word_list in words_per_topic:
        print(f"\n----> Topic <{i}>:")
        for word_sim in word_list:
            print(word_sim)

# Custom Number of Topics.
user_input = input("\nUse Topic Hierarchy to create custom number of topics?\n(yes/[no]) ")
user_input = user_input.strip().lower()
ask_topics = True if user_input in {'y', 'yes'} else False
if ask_topics:
    user_input = input("How many topics? ")
    new_num_topics = int(user_input)
    if new_num_topics >= num_topics:
        print("\nThe Number of Topics need to be smaller than the current Model.")
        print("\nDone.")
        print(f"[{stopwatch.formatted_runtime()}]\n")
        exit()
    else:
        # Upload Full Topic Model.
        print("\nLoading Topic Model with Document and Word Embeddings...")
        model = TopicModel.load(model_id=model.model_id, show_progress=True)
        print("\nDone.")
        print(f"[{stopwatch.formatted_runtime()}]\n")

        # Reduce the Number of Topics.
        print(f"\nUpdating Topic Model with {new_num_topics} topics...")
        model.generate_new_topics(number_topics=new_num_topics, show_progress=True)
        print("\nDone.")
        print(f"[{stopwatch.formatted_runtime()}]\n")

        # Topic's Sizes.
        print("\nTopic's Sizes:")
        topic_sizes = model.top_topics()
        for topic_and_size in topic_sizes:
            print(topic_and_size)
        # Topic's Vocabulary.
        top_n = 15
        print(f"\nTop {top_n} words per topic:")
        words_per_topic = model.all_topics_top_words(top_n)
        for i, word_list in words_per_topic:
            print(f"\n----> Topic <{i}>:")
            for word_sim in word_list:
                print(word_sim)

print("\nDone.")
print(f"[{stopwatch.formatted_runtime()}]\n")
