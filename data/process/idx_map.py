import pandas as pd


def remap_ids_and_save_mappings(data_path, output_path, user_mapping_path, item_mapping_path):
    df = pd.read_csv(data_path)

    user_ids = df["user"].unique()
    item_ids = df["item"].unique()
    user_to_newid = {old_id: new_id for new_id, old_id in enumerate(user_ids)}
    item_to_newid = {old_id: new_id for new_id, old_id in enumerate(item_ids)}

    df["user"] = df["user"].map(user_to_newid)
    df["item"] = df["item"].map(item_to_newid)

    df.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")

    user_mapping_df = pd.DataFrame(list(user_to_newid.items()), columns=['original_id', 'new_id'])
    item_mapping_df = pd.DataFrame(list(item_to_newid.items()), columns=['original_id', 'new_id'])
    user_mapping_df.to_csv(user_mapping_path, index=False)
    item_mapping_df.to_csv(item_mapping_path, index=False)
    print(f"User mapping saved to {user_mapping_path}")
    print(f"Item mapping saved to {item_mapping_path}")


if __name__ == "__main__":
    data_path = "/Users/yyykobe/PycharmProjects/MM_Rec/data/MicroLens-50k/MicroLens-50k_pairs.csv"
    output_path = "/Users/yyykobe/PycharmProjects/MM_Rec/data/MicroLens-50k/MicroLens-50k_pairs_processed.csv"
    user_mapping_path = "user_mapping.csv"
    item_mapping_path = "item_mapping.csv"

    # 调用函数进行处理和保存映射
    remap_ids_and_save_mappings(data_path, output_path, user_mapping_path, item_mapping_path)
