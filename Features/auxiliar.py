#---------------------------------------- Customized plots ----------------------------------------#
#---------------------------------------- Function to create the plot ----------------------------------------#
def customize_plot(df, first_col='', hue='', Size='S'):
##   #get count of tickets on the basis of any specific field against other
   if Size == 'S':
       fig = (10,5)
   elif Size == 'M':
       fig = (15, 7)
   else:
       fig = (20,10)
   plt.figure(figsize=(fig))
   title = hue.capitalize() + " under " + first_col.capitalize()
   
   ax = sns.countplot(x=first_col, hue=hue, data=df, palette=sns.color_palette("Paired"))
   ax.yaxis.set_ticks_position('left')
   ax.xaxis.set_ticks_position('bottom')
   ax.spines['top'].set_visible(False)
   ax.spines['right'].set_visible(False)
   ax.get_xaxis().tick_bottom()
   ax.get_yaxis().tick_left()
   for p in ax.patches:
       if p.get_height() > 50:
          ax.annotate('{:0.0f}'.format(p.get_height()), (p.get_x()+0, p.get_height()+25.5),color='#3274a1',fontsize=10)
   ax.bar_label(ax.containers[0], color='#3274a1',padding=3,fontsize=10)
   plt.title(title)
   plt.xlabel(first_col.capitalize(), fontsize=12)
   plt.ylabel('Ticket Counts', fontsize=12)
   plt.setp(ax.get_xticklabels(), rotation=90)
   plt.legend(loc='upper right',labelspacing=1, title=hue.capitalize(),ncol = 2)
   plt.tight_layout()
   plt.xticks(rotation=45, ha='right')
   plt.savefig(f'C:/Users/RREYESJI/GADM_Projects/Data Analysis/Plots/General/Service Requests/{title}_top_values.png')
   #plt.show()

customize_plot(df_data,'Impact','Urgency','M')
#print(f"Important fields -> \n{important_features}")


#---------------------------------------- Date and time analysis ----------------------------------------#
#extract Year, Month, Hour, Weekday from Created column
df_data['Year'] = pd.DatetimeIndex(df_data['Created']).year
df_data['Month'] = pd.DatetimeIndex(df_data['Created']).month
df_data['Week'] = df_data['Created'].dt.strftime('%Y-%U')
df_data['Hour'] = pd.DatetimeIndex(df_data['Created']).hour
df_data['Day_of_week'] = pd.DatetimeIndex(df_data['Created']).dayofweek
df_data['Day_name'] = pd.DatetimeIndex(df_data['Created']).day_name()
df_data['Day'] = pd.DatetimeIndex(df_data['Created']).day
df_data['Quarter'] = pd.PeriodIndex(df_data['Created'], freq='Q')
#get the count of tickets for years
df_opened_at_year = df_data.groupby(['Year']).size().reset_index(name='Count')
df_opened_at_year = df_opened_at_year.set_index('Year')
df_opened_at_year.sort_values(by='Count', ascending=False)
#get the count of tickets for months
df_opened_at_month = df_data.groupby(['Month']).size().reset_index(name='Count')
df_opened_at_month = df_opened_at_month.set_index('Month')
df_opened_at_month.sort_values(by='Count', ascending=False)
#get the count of tickets for week
df_opened_at_week = df_data.groupby(['Week']).size().reset_index(name='Count')
df_opened_at_week = df_opened_at_week.set_index('Week')
df_opened_at_week.sort_values(by='Count', ascending=False)
#get the count of tickets for day of week
df_opened_at_day_of_week = df_data.groupby(['Day_name']).size().reset_index(name='Count')
df_opened_at_day_of_week = df_opened_at_day_of_week.set_index('Day_name')
df_opened_at_day_of_week.sort_values(by='Count', ascending=False)
#get the count of tickets for hour
df_opened_at_hour = df_data.groupby(['Hour']).size().reset_index(name='Count')
df_opened_at_hour = df_opened_at_hour.set_index('Hour')
df_opened_at_hour.sort_values(by='Count', ascending=False)

#---------------------------------------- Terminal output of date and time analysis ----------------------------------------#
print(f"Opened at year -> \n{df_opened_at_year}")
print(f"Opened at month -> \n{df_opened_at_month}")
print(f"Opened at week -> \n{df_opened_at_week}")
print(f"Opened at day of the week -> \n{df_opened_at_day_of_week}")
print(f"Opened at hour -> \n{df_opened_at_hour}")

#---------------------------------------- Plots of date and time analysis ----------------------------------------#
to_plot = ['df_opened_at_month', 'df_opened_at_week', 'df_opened_at_day_of_week', 'df_opened_at_hour']

# Define titles and labels for each plot
titles = {
    'df_opened_at_month': 'Month & Tickets Count',
    'df_opened_at_week': 'Week & Tickets Count',
    'df_opened_at_day_of_week': 'Day of Week & Tickets Count',
    'df_opened_at_hour': 'Hour & Tickets Count'
}

y_labels = {
    'df_opened_at_month': 'Count of Tickets',
    'df_opened_at_week': 'Count of Tickets',
    'df_opened_at_day_of_week': 'Count of Tickets',
    'df_opened_at_hour': 'Count of Tickets'
}

x_labels = {
    'df_opened_at_month': 'Months',
    'df_opened_at_week': 'Weeks',
    'df_opened_at_day_of_week': 'Days of Week',
    'df_opened_at_hour': 'Hours'
}

# Iterate over the DataFrames and plot them
for df_name in to_plot:
    df = globals()[df_name]  # Get the DataFrame by its name from the global scope
    plt.figure(figsize=(10, 6))
    df.plot()
    plt.xticks(rotation=45)
    plt.title(titles[df_name], fontdict={'fontsize': 20})
    plt.ylabel(y_labels[df_name], fontsize=18)
    plt.xlabel(x_labels[df_name], fontsize=18)
    plt.savefig(f'C:/Users/RREYESJI/GADM_Projects/Data Analysis/Plots/General/Service Requests/{df_name}.png')
    plt.close()  # Close the plot to avoid displaying it in the loop

#---------------------------------------- Grouping ----------------------------------------#
def grouped_month_df(df, extended_cols = []):
    """ Create grouped month df from sub df
    params: df: sub data frame
    return: Sub grouped df
    """
    group_df_name = df.groupby(extended_cols).size().reset_index(name='Ticket Count')
    group_df_name['Ticket Percent'] = round((group_df_name['Ticket Count'] / group_df_name['Ticket Count'].sum()) * 100, 2)
    return group_df_name

#['Number', 'Task type', 'Configuration item', 'Impact', 'Urgency',
#       'Priority', 'Opened', 'Created', 'Opened by', 'Created by', 'State',
#       'Assignment group', 'Assigned to', 'Short description', 'Description',
#       'Work notes', 'Updated', 'Updated by', 'Comments and Work notes',
#       'Requested for']   

#group_df1 = grouped_month_df(df_data, extended_cols=['Company','Assignment group'])
group_df1 = grouped_month_df(df_data, extended_cols=['Configuration item','Assignment group'])

def get_grouped_df(df):
    print(df.sort_values(by=['Ticket Count'], ascending=[False]).head(50).reset_index())

#top_20_companies = list(group_df1.nlargest(20, 'Ticket Count')['Company'].unique())
top_20_groups = list(group_df1.nlargest(20, 'Ticket Count')['Assignment group'].unique())
#print(f"Top 20 Companies ->{top_20_companies}")

for group in top_20_groups:
    print(f"Assignment group: {group}")
    group_df = grouped_month_df(df_data[df_data['Assignment group'] == group], extended_cols=['Configuration item','Short description_Cleaned'])
    group_df.to_excel('SR_ConfigurationItem_by_Short_desc_grouping.xlsx', header=True)
    get_grouped_df(group_df)

#---------------------------------------- Word Clouds ----------------------------------------#
from wordcloud import WordCloud
fig, ax = plt.subplots(1,3, figsize=(15,7.5))
applications = list(group_df1.nlargest(3, 'Ticket Count')['Assignment group'].unique()) #highest activity groups
for app, col in zip(applications, list(range(0,3))):
    corpus = ' '.join(df_data[df_data['Assignment group'] == app]['Short description_Cleaned'])
    ax[col].set_title(f"Word Cloud for {app}")
    ax[col].imshow(WordCloud(
                             height = 200,
                             background_color ='white', 
                             min_font_size = 15).generate(corpus))
    ax[col].axis("off")
#     plt.tick_params(axis='both', which='major', labelsize=12)
plt.tight_layout()
plt.savefig(f'C:/Users/RREYESJI/GADM_Projects/Data Analysis/Plots/General/Service Requests/WordCloud.png')
plt.close()  # Close the plot to avoid displaying it in the loop

#---------------------------------------- Cosine Similarity ----------------------------------------#
#top_20_groups
# Example top_20_groups definition, replace this with your actual logic
df_data['Short description_Cleaned'] = df_data['Short description_Cleaned'].fillna('').astype(str)
top_20_groups = df_data['Assignment group'].value_counts().head(20).index.tolist()
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

