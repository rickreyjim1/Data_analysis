import networkx as nx
import community
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

#---------------------------------------- Cosine Similarity ----------------------------------------#
#top_20_groups
# Example top_20_groups definition, replace this with your actual logic
df_data['short_description_cleaned'] = df_data['short_description_cleaned'].fillna('').astype(str)
top_20_groups = df_data[''].value_counts().head(20).index.tolist()
tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0.1)
tfidf_matrices = []
data_sets = []

for source in top_20_groups:
    source_data = df_data[df_data['Assignment group'] == source].drop_duplicates()
    source_data['Short description_Cleaned'] = source_data['Short description_Cleaned'].fillna('').astype(str)  # Ensure all entries are strings
    data_sets.append(source_data['Short description_Cleaned'])

    try:
        tfidf_matrix = tf.fit_transform(source_data['Short description_Cleaned'])
        tfidf_matrices.append(tfidf_matrix)
    except Exception as e:
        print(f"Error processing group '{source}': {e}")

for source in top_20_groups:
    source_data = df_data[df_data['Assignment group'] == source].drop_duplicates()
    data_sets.append(source_data['Short description_Cleaned'])
    tfidf_matrices.append(tf.fit_transform(
source_data['Short description_Cleaned']))

#find the cosine similarity between rows descriptions
matrix_with_cos_sim = []
for m in tfidf_matrices:
    matrix_with_cos_sim.append(linear_kernel(m, m))

top_n_sentences = []
for cs, t in zip(matrix_with_cos_sim, data_sets):
    no_dups = np.array(t)
    i = 0
    top_frame = []
    for c, z in zip(cs, range(len(cs))):  # Use trange instead of tnrange
        # Create vector of titles
        start_name = pd.Series([no_dups[i]] * len(c)) 
        # Index of all similar titles
        ix_top_n = np.argsort(-c)[0:]
        cos_sim = pd.Series(c[ix_top_n])
        names = pd.Series(no_dups[ix_top_n])
        i += 1
        
        top_frame.append(pd.DataFrame([start_name, names, cos_sim]).transpose())
    
    top_frame = pd.concat(top_frame)
    top_frame.columns = ['desc1', 'desc2', 'cos_sim']
    # Remove the similarities for the same sentences
    top_frame['is_same'] = [bool(i == j) for i, j in zip(top_frame['desc1'], top_frame['desc2'])]
    top_frame = top_frame[top_frame['is_same'] != True]
    top_n_sentences.append(top_frame)

for cs, t in zip(matrix_with_cos_sim, data_sets):
    no_dups = np.array(t)
    i = 0
    top_frame = []
    for c, z in zip(cs, range(len(cs))):
        # Create vector of titles
        start_name = pd.Series([no_dups[i]] * len(c)) 
        # Index of all similar titles
        ix_top_n = np.argsort(-c)[0:]
        cos_sim = pd.Series(c[ix_top_n])
        names = pd.Series(no_dups[ix_top_n])
        i +=1
        
        top_frame.append(pd.DataFrame([start_name, names, cos_sim]).transpose())
    
    top_frame = pd.concat(top_frame)
    top_frame.columns = ['desc1', 'desc2', 'cos_sim']
    # Remove the similarities for the same sentences
    top_frame['is_same'] = [bool(i==j) for i, j in zip(top_frame['desc1'], top_frame['desc2'])]
    top_frame = top_frame[top_frame['is_same'] != True]
    top_n_sentences.append(top_frame)

def create_category_cluster(df_data, category, top_n_sentences):
    appended_data = []
    for index, val in enumerate(category):
        cluster_str = val+"_"
        top_frame = top_n_sentences[index]

        if top_frame.shape[0] > 1:
            edges = list(zip(top_frame['desc1'], top_frame['desc2']))
            weighted_edges = list(zip(top_frame['desc1'], top_frame['desc2'], top_frame['cos_sim']))
            nodes = list(set(top_frame['desc1']).union(set(top_frame['desc2'])))
            G = nx.Graph()
            G.add_nodes_from(nodes)
            G.add_edges_from(edges)
            G.add_weighted_edges_from(weighted_edges)
            
            partition = community.community_louvain.best_partition(G)
            pos = nx.spring_layout(G, dim=2)
            community_id = [partition[node] for node in G.nodes()]

            fig = plt.figure(figsize=(10,10))
            plt.title(f"SR_Cluster Plot For {val}")
            nx.draw(G, pos, edge_color=['silver']*len(G.edges()), cmap=plt.cm.tab20,
                    node_color=community_id, node_size=150)
            plt.savefig(f'C:/Users/RREYESJI/GADM_Projects/Data Analysis/Plots/General/Service Requests/{val}_Cluster_Plot.png')

            title, cluster = [], []
            for i in partition.items():
                cluster_title = cluster_str + str(i[1])
                title.append(i[0])
                cluster.append(cluster_title)
            frame_clust = pd.DataFrame({'Short description_Cleaned': title, 'Cluster': cluster})

            df = df_data[df_data['Assignment group'] == val]
            frame_clust = frame_clust.merge(df[['Number', 'Assignment group', 'Short description_Cleaned']], 
                                            how='left', 
                                            on='Short description_Cleaned')  # Corrected merge

            grouped_mat = frame_clust.groupby(['Number', 'Cluster', 'Assignment group', 'Short description_Cleaned']).size().reset_index(name='Ticket Count')
            grouped_mat.columns = ['Number', 'Cluster', 'Assignment group', 'Short description_Cleaned', 'Ticket Count']
            grouped_mat = grouped_mat.sort_values(by=['Cluster'])
            
            appended_data.append(grouped_mat)

    appended_data = pd.concat(appended_data)
    return appended_data

final_cluster_df = create_category_cluster(df_data, top_20_groups, top_n_sentences)
cluster_df = final_cluster_df[['Number','Cluster']]
cluster_df['Number'] = cluster_df['Number'].astype(str)

#assign clusters to the original df on the basis of Ticket Number
new_cluster_df = pd.merge(df_data, cluster_df,left_on='Number', right_on='Number', how='inner')
print(f"With clusters -> \n{new_cluster_df}")
new_cluster_df.to_excel('SR_with_clusters.xlsx', header=True)