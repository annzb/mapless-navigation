import os.path
import sqlite3
import sys
import yaml


def replace_message_type(ros2_bag_directory_path, old_package, new_package):
    metadata_file_path = os.path.join(ros2_bag_directory_path, 'metadata.yaml')
    bag_name = os.path.basename(ros2_bag_directory_path)
    db3_file_path = os.path.join(ros2_bag_directory_path, f'{bag_name}.db3')
    if not os.path.isfile(metadata_file_path):
        raise ValueError(f'File {metadata_file_path} not found')
    if not os.path.isfile(db3_file_path):
        raise ValueError(f'File {db3_file_path} not found')

    replace_in_metadata_yaml(metadata_file_path, old_package, new_package)

    conn = sqlite3.connect(db3_file_path)
    cursor = conn.cursor()

    # Find entries in the topics table that use the old package for message types
    cursor.execute("SELECT id, type FROM topics WHERE type LIKE ?", (f'{old_package}/%',))
    topics = cursor.fetchall()

    # Check if there are any topics to update
    if not topics:
        print(f"No message types found with package '{old_package}' in '{db3_file_path}'")
        conn.close()
        return

    # Update each message type
    for topic_id, message_type in topics:
        new_message_type = message_type.replace(old_package, new_package)
        cursor.execute("UPDATE topics SET type = ? WHERE id = ?", (new_message_type, topic_id))

    # Commit the changes and close the connection
    conn.commit()
    conn.close()
    print(f"Finished updating message types in '{db3_file_path}'")

def replace_in_metadata_yaml(metadata_file_path, old_package, new_package):
    # Load the YAML file
    with open(metadata_file_path, 'r') as yaml_file:
        metadata = yaml.safe_load(yaml_file)

    # Traverse the YAML structure to find and replace the message types
    for topic in metadata.get('rosbag2_bagfile_information', {}).get('topics_with_message_count', []):
        if 'topic_metadata' in topic and 'type' in topic['topic_metadata']:
            message_type = topic['topic_metadata']['type']
            if old_package in message_type:
                new_message_type = message_type.replace(old_package, new_package)
                topic['topic_metadata']['type'] = new_message_type

    # Write the updated metadata back to the YAML file
    with open(metadata_file_path, 'w') as yaml_file:
        yaml.safe_dump(metadata, yaml_file)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python replace_message_type.py <ros2 bag directory> <old_package> <new_package>")
        sys.exit(1)

    db3_file = sys.argv[1]
    old_package = sys.argv[2]
    new_package = sys.argv[3]

    replace_message_type(db3_file, old_package, new_package)
