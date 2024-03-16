import pandas as pd


def update_titles_with_new_ids(item_mapping_path, titles_path, updated_titles_path):
    item_mapping_df = pd.read_csv(item_mapping_path)
    titles_df = pd.read_csv(titles_path, names=['item', 'title'], header=None)

    item_mapping_df['original_id'] = item_mapping_df['original_id'].astype(str)
    titles_df['item'] = titles_df['item'].astype(str)

    updated_titles_df = pd.merge(titles_df, item_mapping_df, left_on='item', right_on='original_id')

    updated_titles_df = updated_titles_df[['new_id', 'title']].rename(columns={'new_id': 'item'})

    updated_titles_df['item'] = updated_titles_df['item'].astype(int)

    updated_titles_df.to_csv(updated_titles_path, index=False, header=False)
    print(f"Updated titles saved to {updated_titles_path}")


if __name__ == "__main__":
    item_mapping_path = "item_mapping.csv"
    titles_path = "/Users/yyykobe/PycharmProjects/MM_Rec/data/MicroLens-50k/MicroLens-50k_titles.csv"
    updated_titles_path = "/Users/yyykobe/PycharmProjects/MM_Rec/data/MicroLens-50k/MicroLens-50k_titles_processed.csv"

    update_titles_with_new_ids(item_mapping_path, titles_path, updated_titles_path)
