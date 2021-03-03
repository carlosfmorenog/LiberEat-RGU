import os
import pandas as pd
import numpy as np
import dash
import dash_table
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import dash_core_components as dcc
import dash_html_components as html
from plotly.subplots import make_subplots
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from dash.dependencies import Input, Output
from mpl_toolkits.mplot3d import Axes3D
from collections import Counter

def func_label(label0, df_chain, df_location, df_menu, setting):
	df_chain = pd.concat([df_chain['id'], df_chain['name']], axis=1, keys=['chain_id', 'chain_name'])
	df_location = pd.concat([df_location['id'], df_location['chain_id'], df_location['name']], axis=1, keys=['foreign_id', 'chain_id', 'location_name'])
	df_location = pd.merge(df_location, df_chain, on = 'chain_id')
	df_location = df_location.drop('chain_id', 1)
	df_menu = pd.concat([df_menu['id'], df_menu['chain_id'], df_menu['name']], axis=1, keys=['foreign_id', 'chain_id', 'menu_name'])
	df_menu = pd.merge(df_menu, df_chain, on = 'chain_id')
	df_menu = df_menu.drop('chain_id', 1)

	if setting == 32:
		if label0 in df_menu['foreign_id'].unique():
			return_label = df_menu.loc[df_menu['foreign_id'] == label0]['menu_name'].values[0] + ' - ' + df_menu.loc[df_menu['foreign_id'] == label0]['chain_name'].values[0]
			return(return_label)
		else:
			return(label0)
	else:
		if label0 in df_location['foreign_id'].unique():
			return_label = df_location.loc[df_location['foreign_id'] == label0]['location_name'].values[0]
			return(return_label)
		else:
			return(label0)

def zero_to_nan(values):
	"""Replace every 0 with 'nan' and return a copy."""
	return [float('nan') if x==0 else x for x in values]

def plots_tab(fig):
	return dcc.Tab(children = [
		dcc.Graph( figure = fig
		)
	]),

def plots_dash(fig):
	return html.Div([
		dcc.Graph(
			figure = fig
		)
	])

def plots_tick(df_times, position, i_chain):
	df_times = df_times[(df_times['chain_name'] == i_chain)]
	df_times = df_times[['created_at']]

	list_dates = df_times.values.T.tolist()[position]
	for i, j in enumerate(list_dates):
		list_dates[i] = j[:10]
	dict_time = Counter(list_dates)

	x_dates = list(dict_time.keys())
	y_count = list(dict_time.values())

	fig = go.Figure(go.Scatter(
		x = x_dates,
		y = y_count,
	))
	fig.update_layout(
		title = 'Time Series',
		xaxis_tickformat = '%d %B (%a)<br>%Y'
	)
	return plots_dash(fig)

def extra(df_chain, df_location, df_menu, df_maxs, df_complete, setting, i_chain, i_feature):
	df_chain = pd.concat([df_chain['id'], df_chain['name']], axis=1, keys=['chain_id', 'chain_name'])
	df_location = pd.concat([df_location['id'], df_location['chain_id'], df_location['name']], axis=1, keys=['foreign_id', 'chain_id', 'location_name'])
	df_location = pd.merge(df_location, df_chain, on = 'chain_id')
	df_location = df_location.drop('chain_id', 1)
	df_menu = pd.concat([df_menu['id'], df_menu['chain_id'], df_menu['name']], axis=1, keys=['foreign_id', 'chain_id', 'menu_name'])
	df_menu = pd.merge(df_menu, df_chain, on = 'chain_id')
	df_menu = df_menu.drop('chain_id', 1)
	df_maxs = pd.concat([df_maxs['id'], df_maxs['user_id'], df_maxs['type'], df_maxs['foreign_id'], df_maxs['created_at']], axis=1, keys=['id', 'user_id', 'type', 'foreign_id', 'created_at'])
	df_maxs = df_maxs[(df_maxs.type == 31) | (df_maxs.type == 32)]

	df_alls = pd.merge(df_complete, df_maxs, on = 'id')
	df_alls_loc_time = pd.merge(df_alls, df_location, on = 'foreign_id')
	df_alls_men_time = pd.merge(df_alls, df_menu, on = 'foreign_id')
	df_alls_loc = df_alls_loc_time.drop('created_at', 1)
	df_alls_men = df_alls_men_time.drop('created_at', 1)

	if setting == 32:
		if i_chain in df_alls_men['chain_name'].unique():
			men_list = set(df_alls_men.loc[df_alls_men['chain_name'] == i_chain]['menu_name'].tolist())
			return plots_bar(men_list, df_alls_men, i_feature, i_chain, setting), plots_tick(df_alls_men_time, 0, i_chain)
		else:
			return None, None
	else:
		if i_chain in df_alls_loc['chain_name'].unique():
			loc_list = set(df_alls_loc.loc[df_alls_loc['chain_name'] == i_chain]['location_name'].tolist())
			return plots_bar(loc_list, df_alls_loc, i_feature, i_chain, setting), plots_tick(df_alls_loc_time, 0, i_chain)
		else:
			return None, None

def plots_bar(loc_list, df, i_feature, i_chain, setting):
	df = df[(df['chain_name'] == i_chain)]
	if i_feature == 'Gender':
		x_list = ['Women', 'Men', 'Other']
	if i_feature == 'Age':
		x_list = ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81+']
	if i_feature == 'Lifestyle':
		x_list = ['Any', 'Vegan', 'Vegetarian']
	fig = go.Figure()
	fig.add_traces(data = [
		go.Bar(name = str(name), x = x_list, y = obtain_y(name, df, i_feature, setting), 
			text = obtain_y(name, df,  i_feature, setting)) for name in loc_list
		])
	fig.update_traces(textfont_color = 'white', texttemplate = '%{text:.2s}', textposition = 'auto')
	fig.update_layout(
		title = str(i_chain),
		xaxis_tickfont_size = 14,
		xaxis = dict(
			title = str(i_feature),
			titlefont_size = 16,
			tickfont_size = 14,
		),
		yaxis = dict(
			title = 'Amount of users',
			titlefont_size = 16,
			tickfont_size = 14,
		),
		barmode = 'group',
		bargap = 0.15,
		bargroupgap = 0.1,
	)
	return plots_dash(fig)
	
def obtain_y(loc_name, df, i_feature, setting):
	if setting == 31:
		df = df[(df.location_name == loc_name)]
	else:
		df = df[(df.menu_name == loc_name)]
	sum_a = []
	if i_feature == 'Gender':
		sum_a = [0 for x in range(3)]
		for i, row in df.iterrows():
			if row['gender'] == 1.0:
				sum_a[0] = sum_a[0] + 1
			elif row['gender'] == 0.5:
				sum_a[1] = sum_a[1] + 1
			else :
				sum_a[2] = sum_a[2] + 1
		return sum_a
	elif i_feature == 'Lifestyle':
		sum_a = [0 for x in range(3)]
		for i, row in df.iterrows():
			if row['lifestyle'] == 0.0:
				sum_a[0] = sum_a[0] + 1
			elif row['lifestyle'] == 0.5:
				sum_a[1] = sum_a[1] + 1
			else :
				sum_a[2] = sum_a[2] + 1
		return sum_a
	elif i_feature == 'Age':
		sum_a = [0 for x in range(9)]
		for i, row in df.iterrows():
			if 0 <= row['age'] <= 10:
				sum_a[0] = sum_a[0] + 1
			elif 11 <= row['age'] <= 20:
				sum_a[1] = sum_a[1] + 1
			elif 21 <= row['age'] <= 30:
				sum_a[2] = sum_a[2] + 1
			elif 31 <= row['age'] <= 40:
				sum_a[3] = sum_a[3] + 1
			elif 41 <= row['age'] <= 50:
				sum_a[4] = sum_a[4] + 1
			elif 51 <= row['age'] <= 60:
				sum_a[5] = sum_a[5] + 1
			elif 61 <= row['age'] <= 70:
				sum_a[6] = sum_a[6] + 1
			elif 71 <= row['age'] <= 80:
				sum_a[7] = sum_a[7] + 1
			elif row['age'] >= 81 :
				sum_a[8] = sum_a[8] + 1
		sum_a = zero_to_nan(sum_a)
		return sum_a

def plots_radar(centroids, data, target, theta_name):
	radar_list = []
	for i,centroid in enumerate(centroids):
		globals()['radar%s' % i] = go.Figure()
		subdata = data[np.where(target == i)]
		distances = np.zeros(centroid.shape[0])
		for j,subdatapoint in enumerate(subdata):
			for k,feature in enumerate(subdatapoint):
				distances[k] += (abs(feature-centroid[k]))
		df = pd.DataFrame(dict(r = distances))
		df = min_max(df, 'r', 1)
		theta = ['f.' + str(m) for m in list(theta_name)]
		#theta = ['f' + str(m) for m in list(df.index)]
		#df['theta'] = theta
		eval('radar%s' % i).add_trace(go.Scatterpolar(
			r = df['r'],
			theta = theta,
			fill='toself',
		))
		radar_list.append(plots_dash(eval('radar%s' % i)))
		#fig = px.line_polar(df, r = 'r', theta = 'theta', line_close = True)
	return radar_list

def plots_pie(group, ev_str, df_ev2):
	if 'age' in df_ev2.columns:
		del df_ev2['age']
		del df_ev2['lifestyle']
		del df_ev2['gender']
	fig = go.Figure()
	fig.add_traces(data = [go.Pie(
		labels = ev_str,
		values = df_ev2.loc[group].values.tolist(),
		scalegroup = 'one',
		)])
	fig.update_traces(textposition='inside',
		hoverinfo = 'label+value',
		marker = dict(line = dict(color = '#000000', width = 0.5)),
		)
	fig.update_layout(title = 'Group ' +str(group)+ ' Pie',
		uniformtext_minsize = 8,
		uniformtext_mode = 'hide',)
	return dcc.Graph(
		figure = fig
	)

def table_dash(df):
	return dash_table.DataTable(
	data = df.to_dict('records'),
	columns = [{"name": i, "id": i} for i in df.columns],
	fixed_columns = { 'headers': True, 'data': 1 },
	style_table = {
		'maxWidth': '1992px',
	},
	style_header = {'backgroundColor': 'rgb(30, 30, 30)'},
	style_cell = {
		'height': 'auto',
		# all three widths are needed
		'minWidth': '140px', 'width': '140px', 'maxWidth': '140px',
		'whiteSpace': 'normal',
		'backgroundColor': 'rgb(60, 60, 60)',
		'color': 'white',
		'textAlign': 'center',
	},
	style_cell_conditional = [
		{'if': {'column_id': 'id'},
		'minWidth': '30px', 'width': '30px', 'maxWidth': '30px',
		'backgroundColor': 'rgb(30, 30, 30)',},
		{'if': {'column_id': 'index'},
		'minWidth': '30px', 'width': '30px', 'maxWidth': '30px',
		'backgroundColor': 'rgb(30, 30, 30)',},
	]
)

def generate_table(dataframe, max_rows=10):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    ])

def event_type(id_e,df_e,a2_unique,matrix_pro_e,b_unique_str):
	matrix_event = []
	matrix_event2 = []
	matrix_pro_e2 = []
	matrix_pro_e3 = []
	e_matrix = []
	for i, row in df_e.iterrows():
		if int(row['type']) == int(id_e):
			e_matrix.append((row['id'],row['type'],row['foreign_id']))
	df = pd.DataFrame(e_matrix, columns = ['id','type','foreign_id'])
	u = df['id'].nunique()
	f = df['foreign_id'].nunique()
	u_unique = df['id'].unique()
	f_unique = sorted(df['foreign_id'].unique())
	for y in range(u):
		matrix_event.append([0 for x in range(f)])
		matrix_event[y].insert(0, u_unique[y])
	for i,row in df.iterrows():
		matrix_event[np.where(u_unique == row['id'])[0][0]] [f_unique.index(row['foreign_id']) + 1] = matrix_event[np.where(u_unique == row['id'])[0][0]] [f_unique.index(row['foreign_id']) + 1] + 1
	f_unique_str = [str(i) for i in f_unique]
	f_unique_str.insert(0,'id')
	
	matches = sorted(list(set(u_unique) & set(a2_unique))) #compering to list and converting set to list
	unmatches = sorted(list(set(a2_unique) - set(matches)))

	for i in range(len(matrix_event)):
		if matrix_event[i][0] in matches:
			matrix_event2.append(matrix_event[i])

	df_o = pd.DataFrame(matrix_event2, columns = f_unique_str)
	df_o.set_index('id', inplace=True)

	for i in range(len(matrix_pro_e)):
		if matrix_pro_e[i][0] in matches:
			matrix_pro_e2.append(matrix_pro_e[i])

	df_o2 = pd.DataFrame(matrix_pro_e2, columns = b_unique_str)
	df_o2.set_index('id', inplace=True)
	
	for i in range(len(matrix_pro_e)):
		if matrix_pro_e[i][0] in unmatches:
			matrix_pro_e3.append(matrix_pro_e[i])

	df_o3 = pd.DataFrame(matrix_pro_e3, columns = b_unique_str)
	df_o3.set_index('id', inplace=True)

	df_o4 = pd.DataFrame(matrix_event, columns = f_unique_str)
	df_o4.set_index('id', inplace=True)
	return df_o, df_o2, df_o3, df_o4, f_unique_str

def matrix_of_groups(df_mof, mof_str, df_final, len_cen):
	rows_all, columns_all = df_mof.shape
	ev = []
	sum_z = []

	for k in range(len_cen):
		sum_z.append([0 for x in range(3)])
		ev.append([0 for x in range(columns_all)])
		ev[k].insert(0, k)

	for i, row in df_mof.iterrows():
		fla = df_final.loc[i, 'target']
		if 'age' in df_mof.columns:
			if row['age'] != 0:
				sum_z[fla][0] = sum_z[fla][0] + 1
			if row['lifestyle'] != 0:
				sum_z[fla][1] = sum_z[fla][1] + 1
			if row['gender'] != 0:
				sum_z[fla][2] = sum_z[fla][2] + 1
		for j in range(columns_all):
			ev[fla][j+1] = ev[fla][j+1] + row[j]

	if 'age' in df_mof.columns:
		for l in range(len_cen):
			for m in range(3):
				if ev[l][52+m] != 0:
					ev[l][52+m] = float(format(ev[l][52+m] / sum_z[l][m], '.2f'))

	df_ev = pd.DataFrame(ev, columns = mof_str)
	return df_ev

def setting_location():
	return 31

def setting_menu():
	return 32

def invalid():
	raise SystemExit

def menu(ans):
	#--------------------------------------------------------------------------------------------------------------------
	#Menu
	if ans == '31': 
		setting = setting_location()
	else:
		setting = setting_menu()
	return setting

def min_max(df_pro, text_min, posit):
	m_pro_len = len(df_pro.columns)
	x = df_pro[[text_min]].values.astype(float)
	min_max_scaler = preprocessing.MinMaxScaler()
	x_scaled = min_max_scaler.fit_transform(x)
	del(df_pro[text_min])
	df_pro.insert(loc = m_pro_len - posit, column = text_min, value = x_scaled)
	return df_pro

def clustering(range_n_clusters, dfpro, df_completes, i_clusters):
	#Silhouette and clustering
	silhouette = []
	for n_clusters in range_n_clusters:
		clusterer = KMeans(n_clusters = n_clusters, random_state=8)
		cluster_labels = clusterer.fit_predict(dfpro)
		silhouette.append(silhouette_score(dfpro, cluster_labels))

	max_value = max(silhouette)
	list_max_value = [i+2 for i, j in enumerate(silhouette) if j == max_value]

	kmeans = KMeans(n_clusters = list_max_value[0]).fit(dfpro)
	centroids = kmeans.cluster_centers_
	labels_t = kmeans.labels_
	fig_list = plots_radar(centroids, dfpro.values, labels_t, i_clusters)
	len_cen = len(centroids)
	y = KMeans(n_clusters = list_max_value[0]).fit_predict(dfpro)

	#PCA and plotting
	pca = PCA(n_components = 2)
	pca.fit(dfpro)
	dfpro_pca = pca.transform(dfpro)

	finalDf = pd.DataFrame(data = dfpro_pca, columns = ['principal component 1', 'principal component 2'])
	finalDf['target'] = y
	df_completes['target'] = y
	list_df4 = list(map(int,list(dfpro.index)))
	finalDf['id'] = list_df4
	finalDf.set_index('id', inplace=True)
	targets = [x for x in range(len_cen)]
	colors = ['red', 'green', 'blue', 'yellow', 'black', 'orange', 'brown', 'purple', 'pink', 'gray']
	return fig_list, len_cen, finalDf, df_completes, targets, silhouette

def all_the_code(settings, i_chains, i_features, i_clusters):
	#--------------------------------------------------------------------------------------------------------------------
	#Correcting errors
	dir_path = os.path.dirname(os.path.realpath("events.csv"))
	df = pd.read_csv(dir_path + "\\events.csv", encoding='cp1252')
	df_allergen = pd.read_csv(dir_path + "\\allergen_profile.csv", encoding='cp1252')
	df_profile = pd.read_csv(dir_path + "\\profiles.csv", encoding='cp1252')
	df_profile = df_profile[['id', 'user_id', 'dob', 'birth_year', 'gender', 'lifestyle', 'primary', 'active']]
	df_chains = pd.read_csv(dir_path + "\\chains.csv", encoding='cp1252')
	df_locations = pd.read_csv(dir_path + "\\locations.csv", encoding='cp1252')
	df_menus = pd.read_csv(dir_path + "\\menus.csv", encoding='cp1252')

	df['created_at'] = pd.to_datetime(df.created_at)
	df['updated_at'] = pd.to_datetime(df.updated_at)
	df['created_at'] = df['created_at'].dt.strftime('%d/%m/%Y  %H:%M')
	df['updated_at'] = df['updated_at'].dt.strftime('%d/%m/%Y  %H:%M')
	df_profile['dob'] = pd.to_datetime(df_profile.dob)

	test = []
	delete = []

	df_event_drop = df.drop('id', 1)
	df_delete = df_profile

	for i,row in df_delete.iterrows():
		if not row['user_id'] in test:
			test.append(row['user_id'])
		else:
			delete.append(row['id'])
	for i in delete:
		df_delete = df_delete[df_delete.id != i]

	df_max = pd.merge(df_profile, df_event_drop, on = 'user_id')
	df_min = pd.merge(df_delete, df_event_drop, on = 'user_id')

	#--------------------------------------------------------------------------------------------------------------------
	matrix = []
	matrix2 = []
	matrix_pro = []
	matrix_pro2 = []

	h = df['user_id'].nunique()
	w = df['type'].nunique()
	h_unique = df['user_id'].unique()
	w_unique = sorted(df['type'].unique())

	a = df_allergen['profile_id'].nunique()
	b = df_allergen['allergen_id'].nunique()
	a_unique = df_allergen['profile_id'].unique()
	b_unique = sorted(df_allergen['allergen_id'].unique())

	a2 = df_profile['id'].nunique()
	b2 = df_profile['dob'].nunique()
	a2_unique = df_profile['id'].unique()
	b2_unique = df_profile['dob'].unique()

	matches = sorted(list(set(h_unique) & set(a2_unique))) #compering to list and converting set to list
	matches_l = len(matches)

	for y in range(h):
		matrix.append([0 for x in range(w)])
		matrix[y].insert(0,h_unique[y])

	for i,row in df.iterrows():
		matrix[np.where(h_unique == row['user_id'])[0][0]] [w_unique.index(row['type']) + 1] = matrix[np.where(h_unique == row['user_id'])[0][0]] [w_unique.index(row['type']) + 1] + 1

	for c in range(a2):
		matrix_pro.append([0 for x in range(b+3)])
		matrix_pro[c].insert(0,a2_unique[c])

	for i,row in df_allergen.iterrows():
		matrix_pro[np.where(a2_unique == row['profile_id'])[0][0]] [b_unique.index(row['allergen_id']) + 1] = matrix_pro[np.where(a2_unique == row['profile_id'])[0][0]] [b_unique.index(row['allergen_id']) + 1] + 1

	#--------------------------------------------------------------------------------------------------------------------
	m_pro_len = len(matrix_pro[0])
	for i, row in df_profile.iterrows():
		if not pd.isnull(row['dob']):
			if isinstance(row['dob'], pd.Timestamp):
				row_len = 4
			else :
				row_len = len(row['dob'])
		if str(row['gender']).lower() == 'male':
			matrix_pro[i][m_pro_len-1] = 0.5
		elif str(row['gender']).lower() == 'female':
			matrix_pro[i][m_pro_len-1] = 1
		else:
			matrix_pro[i][m_pro_len-1] = 0

		if str(row['lifestyle']).lower() == 'vegan':
			matrix_pro[i][m_pro_len-2] = 0.5
		elif str(row['lifestyle']).lower() == 'vegetarian':
			matrix_pro[i][m_pro_len-2] = 1
		else:
			matrix_pro[i][m_pro_len-2] = 0


		if pd.isnull(row['birth_year']):
			if pd.isnull(row['dob']):
				matrix_pro[i][m_pro_len-3] = 0
			elif (2019 - int(str(row['dob'])[row_len-4:row_len])) > 8 and (2019 - int(str(row['dob'])[row_len-4:row_len])) < 85:
				matrix_pro[i][m_pro_len-3] = 2019 - int(str(row['dob'])[row_len-4:row_len])
			else:
				matrix_pro[i][m_pro_len-3] = 0
		elif (2019 - int(row['birth_year'])) > 8 and (2019 - int(row['birth_year'])) < 85:
			matrix_pro[i][m_pro_len-3] = (2019 - int(row['birth_year']))
		else:
			matrix_pro[i][m_pro_len-3] = 0

	w_unique_str = [str(i) for i in w_unique]
	w_unique_str.insert(0,'user_id')
	df2 = pd.DataFrame(matrix, columns = w_unique_str)
	df2.set_index('user_id', inplace=True)

	b_unique_str = [str(i) for i in b_unique]
	b_unique_str.insert(0,'id')
	b_unique_str.append('age')
	b_unique_str.append('lifestyle')
	b_unique_str.append('gender')
	df_completes = pd.DataFrame(matrix_pro, columns = b_unique_str)
	dfpro3 = pd.DataFrame(matrix_pro, columns = b_unique_str)
	dfpro3.set_index('id', inplace=True)

	dfpro3 = min_max(dfpro3, 'age', 3)

	dfpro4 = dfpro3[dfpro3.age != 0]
	dfpro5 = dfpro3[['age', 'lifestyle', 'gender']]
	dfpro6 = dfpro4[['age', 'lifestyle', 'gender']]
	range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
	silhouette_avg3 = []
	silhouette_avg4 = []

	#--------------------------------------------------------------------------------------------------------------------
	#Silhouette and clustering & PCA and plotting
	a_clusters = [z for z in dfpro3.columns]
	com_clusters = sorted(list(set(a_clusters) & set(i_clusters)))
	fig_list, len_cen, finalDf, df_completes, targets, silhouette_avg3 = clustering(range_n_clusters, dfpro3[com_clusters], df_completes, i_clusters)

	#--------------------------------------------------------------------------------------------------------------------
	#Multiplying matrices and Matrix of groups
	df_event, df_prof, df_unmatch, df_all_events, e_unique_str = event_type(settings, df_max, a2_unique, matrix_pro, b_unique_str)

	df_event.to_csv('df_event.csv')
	df_prof.to_csv('df_prof.csv')
	df_unmatch.to_csv('df_unmatch.csv')
	finalDf.to_csv('finalDf.csv')

	with open('e_unique_str.txt', 'w') as f:
		for item in e_unique_str:
			f.write("%s\n" % func_label(item, df_chains, df_locations, df_menus, settings))

	with open('m_pro_len.txt', 'w') as f:
		f.write('%d\n' % m_pro_len)
		f.write('%d\n' % settings)

	df_ev = matrix_of_groups(df_event, e_unique_str, finalDf, len_cen)
	df_pr = matrix_of_groups(df_prof, b_unique_str, finalDf, len_cen)
	df_ev.set_index('id', inplace=True)
	df_pr.set_index('id', inplace=True)
	e_unique_str = e_unique_str[1:]
	b_unique_str = b_unique_str[1:]

	array1 = df_pr.values
	array1T = array1.T
	array2 = df_ev.values
	array2T = array2.T
	multi_dot_array = array1T.dot(array2)
	df_multi = pd.DataFrame(data = multi_dot_array, columns = e_unique_str)

	fig3, fig4 = extra(df_chains, df_locations, df_menus, df_max, df_completes, settings, i_chains, i_features)
	
	return finalDf, df_ev, df_pr, df_multi, e_unique_str, b_unique_str, range_n_clusters, silhouette_avg3, targets, fig3, fig_list, len_cen, fig4

#--------------------------------------------------------------------------------------------------------------------
#Lets try dash
#app.css.append_css({'external_url': 'https://codepen.io/amyoshino/pen/jzXypZ.css'})
#app.css.append_css({"external_url": "https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css"})
#external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
external_stylesheets = ['https://codepen.io/amyoshino/pen/jzXypZ.css']

app = dash.Dash(__name__, external_stylesheets = external_stylesheets)

dash_colors = {
	'background': '#111111',
	'text': '#7FDBFF'
}

app.layout = html.Div([
	html.Div([
		html.H1(
			children = 'Liber Eat - Dashboard',
			style= {
				'textAlign': 'left',
				#'color': dash_colors['text']
			},
		),
		html.Label('Make A Choice'),
	], className = "row"),


	html.Div([
		html.Div([
			dcc.Dropdown(
				id = 'i_chain',
				options = [
					{'label': u'Harvester', 'value': 'Harvester'},
					{'label': u'Ember Inns', 'value': 'Ember Inns'},
					{'label': u'Toby Carvery', 'value': 'Toby Carvery'},
					{'label': u'Vintage Inns', 'value': 'Vintage Inns'},
					{'label': u'Crown Carveries', 'value': 'Crown Carveries'},
					{'label': u'Miller & Carter', 'value': 'Miller & Carter'},
					{'label': u'All Bar One', 'value': 'All Bar One'},
					{'label': u'Stonehouse', 'value': 'Stonehouse'},
					{'label': u'Browns', 'value': 'Browns'},
					{'label': u'Nicholson\'s', 'value': 'Nicholson\'s'},
					{'label': u'PCP', 'value': 'PCP'},
					{'label': u'Son of Steak', 'value': 'Son of Steak'},
					{'label': u'Sizzling Pubs', 'value': 'Sizzling Pubs'},
					{'label': u'O\'Neill\'s', 'value': 'O\'Neill\'s'},
					{'label': u'Innkeeper\'s Lodge', 'value': 'Innkeeper\'s Lodge'},
					{'label': u'Orchid Pubs', 'value': 'Orchid Pubs'},
					{'label': u'Castle - Work Rest & Stretch', 'value': 'Castle - Work Rest & Stretch'},
					{'label': u'Fuzzy Ed\'s', 'value': 'Fuzzy Ed\'s'},
					{'label': u'PKB', 'value': 'PKB'},
					{'label': u'Chicken Society', 'value': 'Chicken Society'},
					{'label': u'High Street', 'value': 'High Street'},
					{'label': u'Suburban', 'value': 'Suburban'},
					{'label': u'Harvester 2020', 'value': 'Harvester 2020'},
					{'label': u'Suburban Innovation', 'value': 'Suburban Innovation'},
					{'label': u'Castle - Urban Adventurer', 'value': 'Castle - Urban Adventurer'},
					{'label': u'Oak Tree', 'value': 'Oak Tree'},
					{'label': u'Heritage', 'value': 'Heritage'},
					{'label': u'PCP Mandarin', 'value': 'PCP Mandarin'},
					{'label': u'George and Co', 'value': 'George and Co'},
					{'label': u'Very Vintage', 'value': 'Very Vintage'},
					{'label': u'Pret A Manger', 'value': 'Pret A Manger'},
					{'label': u'Veggie Pret', 'value': 'Veggie Pret'},
				],
				placeholder = "Choose Chain",
				value = ' ',
			),
			dcc.Dropdown(
				id = 'i_dropdown',
				options = [
					{'label': u'Location', 'value': '31'},
					{'label': u'Menu', 'value': '32'},
				],
				placeholder = "Loc / Menu",
				value = ' ',
			),
			dcc.Dropdown(
				id = 'i_feature',
				options = [
					{'label': u'Age', 'value': 'Age'},
					{'label': u'Lifestyle', 'value': 'Lifestyle'},
					{'label': u'Gender', 'value': 'Gender'},
				],
				placeholder = "Choose Feature",
				value = ' ',
			),
			html.Div(id = 'chain'),
			html.Div(id = 'dropdown'),
			html.Div(id = 'feature'),
		], className = "three columns"),


		html.Div([
			dcc.Dropdown(
				id = 'i_cluster',
				options = [
					{'label': u'Age', 'value': 'age'},
					{'label': u'Lifestyle', 'value': 'lifestyle'},
					{'label': u'Gender', 'value': 'gender'},
					{'label': u'Celery & Celeriac (1)', 'value': '1'},
					{'label': u'Gluten & Wheat (2)', 'value': '2'},
					{'label': u'Eggs (4)', 'value': '4'},
					{'label': u'Fish (5)', 'value': '5'},
					{'label': u'Lupin (6)', 'value': '6'},
					{'label': u'Milk (7)', 'value': '7'},
					{'label': u'Mustard (9)', 'value': '9'},
					{'label': u'Tree Nuts (10)', 'value': '10'},
					{'label': u'Peanuts (11)', 'value': '11'},
					{'label': u'Sesame Seeds (12)', 'value': '12'},
					{'label': u'Soya (13)', 'value': '13'},
					{'label': u'Sulphites & Sulphur Dioxide (14)', 'value': '14'},
					{'label': u'Gelatine (15)', 'value': '15'},
					{'label': u'MSG (16)', 'value': '16'},
					{'label': u'Food Colourings (17)', 'value': '17'},
					{'label': u'Sodium Benzoate (18)', 'value': '18'},
					{'label': u'Nitrates & Nitrites (Preservatives) (19)', 'value': '19'},
					{'label': u'Sweetners (Selected) (20)', 'value': '20'},
					{'label': u'Yeast (21)', 'value': '21'},
					{'label': u'Palm Oil (22)', 'value': '22'},
					{'label': u'Barley (23)', 'value': '23'},
					{'label': u'Oats (24)', 'value': '24'},
					{'label': u'Rye (25)', 'value': '25'},
					{'label': u'Wheat (26)', 'value': '26'},
					{'label': u'May contain Gluten (27)', 'value': '27'},
					{'label': u'May contain Wheat (28)', 'value': '28'},
					{'label': u'Almonds (34)', 'value': '34'},
					{'label': u'Brazil (35)', 'value': '35'},
					{'label': u'Cashew (36)', 'value': '36'},
					{'label': u'Hazelnut (37)', 'value': '37'},
					{'label': u'Macadamia (38)', 'value': '38'},
					{'label': u'Pecan (39)', 'value': '39'},
					{'label': u'Pistachio (40)', 'value': '40'},
					{'label': u'Walnut (41)', 'value': '41'},
					{'label': u'May contain nuts (42)', 'value': '42'},
					{'label': u'May contain Peanuts (44)', 'value': '44'},
					{'label': u'May contain Sulphites / Sulphur Dioxides (48)', 'value': '48'},
					{'label': u'Buffalo Milk (50)', 'value': '50'},
					{'label': u'Goat Milk (51)', 'value': '51'},
					{'label': u'Sheep Milk (52)', 'value': '52'},
					{'label': u'May contain Milk (53)', 'value': '53'},
					{'label': u'May contain Fish (56)', 'value': '56'},
					{'label': u'May contain Soya (58)', 'value': '58'},
					{'label': u'May contain Egg (60)', 'value': '60'},
					{'label': u'May contain Celery (63)', 'value': '63'},
					{'label': u'May contain Lupin (66)', 'value': '66'},
					{'label': u'May contain Sesame (68)', 'value': '68'},
					{'label': u'May contain Mustard (70)', 'value': '70'},
					{'label': u'E102 (tartrazine) (71)', 'value': '71'},
					{'label': u'E104 (quinoline yellow) (72)', 'value': '72'},
					{'label': u'E110 (sunset yellow FCF) (73)', 'value': '73'},
					{'label': u'E122 (carmoisine) (74)', 'value': '74'},
					{'label': u'E124 (ponceau 4R) (75)', 'value': '75'},
					{'label': u'E129 (allura red) (76)', 'value': '76'},
					{'label': u'Nitrates (77)', 'value': '77'},
					{'label': u'Nitrites (78)', 'value': '78'},
					{'label': u'Acesulfame (79)', 'value': '79'},
					{'label': u'Aspartame (80)', 'value': '80'},
					{'label': u'Saccharin (81)', 'value': '81'},
					{'label': u'Sorbitol (82)', 'value': '82'},
					{'label': u'Sucralose (83)', 'value': '83'},
					{'label': u'Stevia (84)', 'value': '84'},
					{'label': u'Xylitol (85)', 'value': '85'},
					{'label': u'Erythritol (86)', 'value': '86'},
					{'label': u'Isomalt (87)', 'value': '87'},
					{'label': u'Lactitol (88)', 'value': '88'},
					{'label': u'Maltitol (89)', 'value': '89'},
					{'label': u'Cyclamic acid (90)', 'value': '90'},
					{'label': u'Chestnuts (92)', 'value': '92'},
					{'label': u'Pine Nuts & Pine Kernels (93)', 'value': '93'},
					{'label': u'Shellfish (125)', 'value': '125'},
					{'label': u'Molluscs (127)', 'value': '127'},
					{'label': u'May contain Molluscs (128)', 'value': '128'},
					{'label': u'Crustaceans (130)', 'value': '130'},
					{'label': u'May contain Crustaceans (131)', 'value': '131'},
					{'label': u'May contain Shellfish (132)', 'value': '132'},
					{'label': u'E133 (brilliant blue FCF) (133)', 'value': '133'},
					{'label': u'Cow Milk (134)', 'value': '134'},
					{'label': u'Gluten Free Oats (135)', 'value': '135'},
				],
				value = ['age', 'lifestyle', 'gender', '1', '2', '4', '5', '6', '7', '9', '10', '11', '12', '13', '14', '15', 
				'16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '34', '35', '36', '37', '38', 
				'39', '40', '41', '42', '44', '48', '50', '51', '52', '53', '56', '58', '60', '63', '66', '68', '70', '71', 
				'72', '73', '74', '75', '76', '77', '78', '79', '80', '81', '82', '83', '84', '85', '86', '87', '88', '89', 
				'90', '92', '93', '125', '127', '128', '130', '131', '132', '133', '134','135'],
				multi = True,
			),
			#html.Button('Run Cluster', id = 'btn_runcluster', n_clicks = 0),
			html.Div(id = 'runcluster'),
		], className = "nine columns"),
	], className = "row"),


	html.Br(), html.Br(),
	html.Div([
		html.Div([
			html.Button('Run Graphs', id='btn_rungraphs', n_clicks=0, 
				style = {
				'margin-left' : 25
				}
			),
		], className = "offset-by-five columns"),
	], className = "row"),
	html.Br(), html.Br(),


	html.Div([
		html.Div([
			html.Div(id = 'rungraphs'),
		]),

		html.Div([
			html.Div(id = 'graph_time'),
		]),

	], className = "row"),
	html.Br(), html.Br(),


	html.Div([
		html.Div([
			dcc.Tabs(
			parent_className='custom-tabs',
			className='custom-tabs-container',
			children=[
				dcc.Tab(label='Cluster', id = 'PCA_graph', className='custom-tab', selected_className='custom-tab--selected'),
				dcc.Tab(label='Group 0', id = 'output_cluster0', className='custom-tab', selected_className='custom-tab--selected'),
				dcc.Tab(label='Group 1', id = 'output_cluster1', className='custom-tab', selected_className='custom-tab--selected'),
				dcc.Tab(label='Group 2', id = 'output_cluster2', className='custom-tab', selected_className='custom-tab--selected'),
				dcc.Tab(label='Group 3', id = 'output_cluster3', className='custom-tab', selected_className='custom-tab--selected'),
				dcc.Tab(label='Group 4', id = 'output_cluster4', className='custom-tab', selected_className='custom-tab--selected'),
				dcc.Tab(label='Group 5', id = 'output_cluster5', className='custom-tab', selected_className='custom-tab--selected'),
				dcc.Tab(label='Group 6', id = 'output_cluster6', className='custom-tab', selected_className='custom-tab--selected'),
				dcc.Tab(label='Group 7', id = 'output_cluster7', className='custom-tab', selected_className='custom-tab--selected'),
				dcc.Tab(label='Group 8', id = 'output_cluster8', className='custom-tab', selected_className='custom-tab--selected'),
				dcc.Tab(label='Group 9', id = 'output_cluster9', className='custom-tab', selected_className='custom-tab--selected'),
				dcc.Tab(label='Group 10', id = 'output_cluster10', className='custom-tab', selected_className='custom-tab--selected'),
				dcc.Tab(label='Group 11', id = 'output_cluster11', className='custom-tab', selected_className='custom-tab--selected'),
			]),
		]),
	], className = "row"),
], className='ten columns offset-by-one')
"""
@app.callback(
	Output('runcluster', 'children'),
	[Input('btn_runcluster', 'n_clicks'),
	Input('i_cluster', 'value'),])
def displayClick(n_clicks, value):
	return None
	#return 'You have selected: {}'.format(value)

@app.callback(
	Output('chain', 'children'),
	[Input('i_chain', 'value')])
def update_output(value):
	if value != ' ' and  value != None:
		return None
		#return 'You have selected the chain: {}'.format(value)

@app.callback(
	Output('dropdown', 'children'),
	[Input('i_dropdown', 'value')])
def update_output(value):
	if value != ' ' and  value != None:
		return None
		#return 'You have selected: {}'.format(value)

@app.callback(
	Output('feature', 'children'),
	[Input('i_feature', 'value')])
def update_output(value):
	if value != ' ' and  value != None:
		return None
		#return 'You have selected the feature: {}'.format(value)
"""
@app.callback(
	[Output('rungraphs', 'children'), Output('PCA_graph', 'children'), Output('output_cluster0', 'children'), Output('output_cluster1', 'children'), Output('output_cluster2', 'children'), Output('output_cluster3', 'children'), Output('output_cluster4', 'children'), 
	Output('output_cluster5', 'children'), Output('output_cluster6', 'children'), Output('output_cluster7', 'children'), Output('output_cluster8', 'children'), Output('output_cluster9', 'children'), Output('output_cluster10', 'children'), Output('output_cluster11', 'children'), 
	Output('graph_time', 'children')],
	[Input('btn_rungraphs', 'n_clicks'), Input('i_dropdown', 'value'), Input('i_chain', 'value'), Input('i_feature', 'value'), Input('i_cluster', 'value'),])
def display_value(btn1, i_dropdown, i_chain, i_feature, i_cluster):
	changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
	if 'btn_rungraphs' in changed_id:
		if i_dropdown != ' ' and  i_dropdown != None and i_chain != ' ' and  i_chain != None and i_feature != ' ' and  i_feature != None:
			setting = menu(i_dropdown)
			finalDf, df_ev, df_pr, df_multi, ev_str, pr_str, range_n_clusters, silhouette, targets, fig3, fig_list, len_cen, fig4 = all_the_code(setting, i_chain, i_feature, i_cluster)
			
			"""
			df_ev2 = df_ev.reset_index()
			df_pr2 = df_pr.reset_index()
			df_multi = df_multi.reset_index()

			fig = go.Figure(data = [go.Scatter(x = range_n_clusters, y = silhouette)])
			fig.update_layout(title = 'Silhouette Score Elbow for KMeans Clustering (Everything)',
				xaxis_title = 'k',
				yaxis_title = 'silhouette score')
			"""

			fig2 = go.Figure()
			for target in targets:
				indicesToKeep = finalDf['target'] == target
				fig2.add_trace(go.Scatter(
					x = finalDf.loc[indicesToKeep, 'principal component 1'], 
					y = finalDf.loc[indicesToKeep, 'principal component 2'], 
					name = str(target),
					))
			fig2.update_traces(mode = 'markers')
			fig2.update_layout(title = 'KMeans Clustering PCA',
				xaxis_title = 'PCA 1',
				yaxis_title = 'PCA 2',)
				#margin = {'l': 20, 'r': 10, 'b': 20, 't': 10}
			if len_cen == 1:
				return [fig3, plots_dash(fig2), fig_list[0], None, None, None, None, None, None, None, None, None, None, None, fig4]
			elif len_cen == 2:
				return [fig3, plots_dash(fig2), fig_list[0], fig_list[1], None, None, None, None, None, None, None, None, None, None, fig4]
			elif len_cen == 3:
				return [fig3, plots_dash(fig2), fig_list[0], fig_list[1], fig_list[2], None, None, None, None, None, None, None, None, None, fig4]
			elif len_cen == 4:
				return [fig3, plots_dash(fig2), fig_list[0], fig_list[1], fig_list[2], fig_list[3], None, None, None, None, None, None, None, None, fig4]
			elif len_cen == 5:
				return [fig3, plots_dash(fig2), fig_list[0], fig_list[1], fig_list[2], fig_list[3], fig_list[4], None, None, None, None, None, None, None, fig4]
			elif len_cen == 6:
				return [fig3, plots_dash(fig2), fig_list[0], fig_list[1], fig_list[2], fig_list[3], fig_list[4], fig_list[5], None, None, None, None, None, None, fig4]
			elif len_cen == 7:
				return [fig3, plots_dash(fig2), fig_list[0], fig_list[1], fig_list[2], fig_list[3], fig_list[4], fig_list[5], fig_list[6], None, None, None, None, None, fig4]
			elif len_cen == 8:
				return [fig3, plots_dash(fig2), fig_list[0], fig_list[1], fig_list[2], fig_list[3], fig_list[4], fig_list[5], fig_list[6], fig_list[7], None, None, None, None, fig4]
			elif len_cen == 9:
				return [fig3, plots_dash(fig2), fig_list[0], fig_list[1], fig_list[2], fig_list[3], fig_list[4], fig_list[5], fig_list[6], fig_list[7], fig_list[8], None, None, None, fig4]
			elif len_cen == 10:
				return [fig3, plots_dash(fig2), fig_list[0], fig_list[1], fig_list[2], fig_list[3], fig_list[4], fig_list[5], fig_list[6], fig_list[7], fig_list[8], fig_list[9], None, None, fig4]
			elif len_cen == 11:
				return [fig3, plots_dash(fig2), fig_list[0], fig_list[1], fig_list[2], fig_list[3], fig_list[4], fig_list[5], fig_list[6], fig_list[7], fig_list[8], fig_list[9], fig_list[10], None, fig4]
			elif len_cen == 12:
				return [fig3, plots_dash(fig2), fig_list[0], fig_list[1], fig_list[2], fig_list[3], fig_list[4], fig_list[5], fig_list[6], fig_list[7], fig_list[8], fig_list[9], fig_list[10], fig_list[11], fig4]
		return [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]
	return [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]

if __name__ == '__main__':
	app.run_server(port = 5050)