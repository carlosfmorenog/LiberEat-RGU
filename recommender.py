import os
import pandas as pd
import numpy as np
import dash
import dash_table
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import dash_core_components as dcc
import dash_html_components as html
from sklearn import preprocessing
from dash.dependencies import Output, State, Input

def recomm(sc_input, list_un, df_unmatch2, multi_dot_array, finalDf, e_unique_str, num_recomm):
	sc_input = int(sc_input)
	if sc_input in list_un :
		prediction = list(map(float,df_unmatch2.loc[sc_input].values.dot(multi_dot_array)))
		s_r = 'The user ' + str(sc_input) + ' it is part of the group ' + str(int(finalDf.loc[sc_input]['target'])) + '\r\n'
		s_r1 = 'RECOMENDATIONS:' + '\r\n'
		s_r_l = []
		for i in range(num_recomm):
			m = max(prediction)
			predic_list = [i for i, j in enumerate(prediction) if j == m]
			#print(e_unique_str[predic_list[0] + 1])
			s_r_l.append(e_unique_str[predic_list[0] + 1] + '\r\n')
			prediction[predic_list[0]] = -1
		if num_recomm == 1:
			return(s_r, s_r1, s_r_l[0], None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None)
		elif num_recomm == 2:
			return(s_r, s_r1, s_r_l[0], s_r_l[1], None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None)
		elif num_recomm == 3:
			return(s_r, s_r1, s_r_l[0], s_r_l[1], s_r_l[2], None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None)
		elif num_recomm == 4:
			return(s_r, s_r1, s_r_l[0], s_r_l[1], s_r_l[2], s_r_l[3], None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None)
		elif num_recomm == 5:
			return(s_r, s_r1, s_r_l[0], s_r_l[1], s_r_l[2], s_r_l[3], s_r_l[4], None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None)
		elif num_recomm == 6:
			return(s_r, s_r1, s_r_l[0], s_r_l[1], s_r_l[2], s_r_l[3], s_r_l[4], s_r_l[5], None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None)
		elif num_recomm == 7:
			return(s_r, s_r1, s_r_l[0], s_r_l[1], s_r_l[2], s_r_l[3], s_r_l[4], s_r_l[5], s_r_l[6], None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None)
		elif num_recomm == 8:
			return(s_r, s_r1, s_r_l[0], s_r_l[1], s_r_l[2], s_r_l[3], s_r_l[4], s_r_l[5], s_r_l[6], s_r_l[7], None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None)
		elif num_recomm == 9:
			return(s_r, s_r1, s_r_l[0], s_r_l[1], s_r_l[2], s_r_l[3], s_r_l[4], s_r_l[5], s_r_l[6], s_r_l[7], s_r_l[8], None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None)
		elif num_recomm == 10:
			return(s_r, s_r1, s_r_l[0], s_r_l[1], s_r_l[2], s_r_l[3], s_r_l[4], s_r_l[5], s_r_l[6], s_r_l[7], s_r_l[8], s_r_l[9], None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None)
		elif num_recomm == 11:
			return(s_r, s_r1, s_r_l[0], s_r_l[1], s_r_l[2], s_r_l[3], s_r_l[4], s_r_l[5], s_r_l[6], s_r_l[7], s_r_l[8], s_r_l[9], s_r_l[10], None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None)
		elif num_recomm == 12:
			return(s_r, s_r1, s_r_l[0], s_r_l[1], s_r_l[2], s_r_l[3], s_r_l[4], s_r_l[5], s_r_l[6], s_r_l[7], s_r_l[8], s_r_l[9], s_r_l[10], s_r_l[11], None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None)
		elif num_recomm == 13:
			return(s_r, s_r1, s_r_l[0], s_r_l[1], s_r_l[2], s_r_l[3], s_r_l[4], s_r_l[5], s_r_l[6], s_r_l[7], s_r_l[8], s_r_l[9], s_r_l[10], s_r_l[11], s_r_l[12], None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None)
		elif num_recomm == 14:
			return(s_r, s_r1, s_r_l[0], s_r_l[1], s_r_l[2], s_r_l[3], s_r_l[4], s_r_l[5], s_r_l[6], s_r_l[7], s_r_l[8], s_r_l[9], s_r_l[10], s_r_l[11], s_r_l[12], s_r_l[13], None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None)
		elif num_recomm == 15:
			return(s_r, s_r1, s_r_l[0], s_r_l[1], s_r_l[2], s_r_l[3], s_r_l[4], s_r_l[5], s_r_l[6], s_r_l[7], s_r_l[8], s_r_l[9], s_r_l[10], s_r_l[11], s_r_l[12], s_r_l[13], s_r_l[14], None, None, None, None, None, None, None, None, None, None, None, None, None, None, None)
		elif num_recomm == 16:
			return(s_r, s_r1, s_r_l[0], s_r_l[1], s_r_l[2], s_r_l[3], s_r_l[4], s_r_l[5], s_r_l[6], s_r_l[7], s_r_l[8], s_r_l[9], s_r_l[10], s_r_l[11], s_r_l[12], s_r_l[13], s_r_l[14], s_r_l[15], None, None, None, None, None, None, None, None, None, None, None, None, None, None)
		elif num_recomm == 17:
			return(s_r, s_r1, s_r_l[0], s_r_l[1], s_r_l[2], s_r_l[3], s_r_l[4], s_r_l[5], s_r_l[6], s_r_l[7], s_r_l[8], s_r_l[9], s_r_l[10], s_r_l[11], s_r_l[12], s_r_l[13], s_r_l[14], s_r_l[15], s_r_l[16], None, None, None, None, None, None, None, None, None, None, None, None, None)
		elif num_recomm == 18:
			return(s_r, s_r1, s_r_l[0], s_r_l[1], s_r_l[2], s_r_l[3], s_r_l[4], s_r_l[5], s_r_l[6], s_r_l[7], s_r_l[8], s_r_l[9], s_r_l[10], s_r_l[11], s_r_l[12], s_r_l[13], s_r_l[14], s_r_l[15], s_r_l[16], s_r_l[17], None, None, None, None, None, None, None, None, None, None, None, None)
		elif num_recomm == 19:
			return(s_r, s_r1, s_r_l[0], s_r_l[1], s_r_l[2], s_r_l[3], s_r_l[4], s_r_l[5], s_r_l[6], s_r_l[7], s_r_l[8], s_r_l[9], s_r_l[10], s_r_l[11], s_r_l[12], s_r_l[13], s_r_l[14], s_r_l[15], s_r_l[16], s_r_l[17], s_r_l[18], None, None, None, None, None, None, None, None, None, None, None)
		elif num_recomm == 20:
			return(s_r, s_r1, s_r_l[0], s_r_l[1], s_r_l[2], s_r_l[3], s_r_l[4], s_r_l[5], s_r_l[6], s_r_l[7], s_r_l[8], s_r_l[9], s_r_l[10], s_r_l[11], s_r_l[12], s_r_l[13], s_r_l[14], s_r_l[15], s_r_l[16], s_r_l[17], s_r_l[18], s_r_l[19], None, None, None, None, None, None, None, None, None, None)
		elif num_recomm == 21:
			return(s_r, s_r1, s_r_l[0], s_r_l[1], s_r_l[2], s_r_l[3], s_r_l[4], s_r_l[5], s_r_l[6], s_r_l[7], s_r_l[8], s_r_l[9], s_r_l[10], s_r_l[11], s_r_l[12], s_r_l[13], s_r_l[14], s_r_l[15], s_r_l[16], s_r_l[17], s_r_l[18], s_r_l[19], s_r_l[20], None, None, None, None, None, None, None, None, None)
		elif num_recomm == 22:
			return(s_r, s_r1, s_r_l[0], s_r_l[1], s_r_l[2], s_r_l[3], s_r_l[4], s_r_l[5], s_r_l[6], s_r_l[7], s_r_l[8], s_r_l[9], s_r_l[10], s_r_l[11], s_r_l[12], s_r_l[13], s_r_l[14], s_r_l[15], s_r_l[16], s_r_l[17], s_r_l[18], s_r_l[19], s_r_l[20], s_r_l[21], None, None, None, None, None, None, None, None)
		elif num_recomm == 23:
			return(s_r, s_r1, s_r_l[0], s_r_l[1], s_r_l[2], s_r_l[3], s_r_l[4], s_r_l[5], s_r_l[6], s_r_l[7], s_r_l[8], s_r_l[9], s_r_l[10], s_r_l[11], s_r_l[12], s_r_l[13], s_r_l[14], s_r_l[15], s_r_l[16], s_r_l[17], s_r_l[18], s_r_l[19], s_r_l[20], s_r_l[21], s_r_l[22], None, None, None, None, None, None, None)
		elif num_recomm == 24:
			return(s_r, s_r1, s_r_l[0], s_r_l[1], s_r_l[2], s_r_l[3], s_r_l[4], s_r_l[5], s_r_l[6], s_r_l[7], s_r_l[8], s_r_l[9], s_r_l[10], s_r_l[11], s_r_l[12], s_r_l[13], s_r_l[14], s_r_l[15], s_r_l[16], s_r_l[17], s_r_l[18], s_r_l[19], s_r_l[20], s_r_l[21], s_r_l[22], s_r_l[23], None, None, None, None, None, None)
		elif num_recomm == 25:
			return(s_r, s_r1, s_r_l[0], s_r_l[1], s_r_l[2], s_r_l[3], s_r_l[4], s_r_l[5], s_r_l[6], s_r_l[7], s_r_l[8], s_r_l[9], s_r_l[10], s_r_l[11], s_r_l[12], s_r_l[13], s_r_l[14], s_r_l[15], s_r_l[16], s_r_l[17], s_r_l[18], s_r_l[19], s_r_l[20], s_r_l[21], s_r_l[22], s_r_l[23], s_r_l[24], None, None, None, None, None)
		elif num_recomm == 26:
			return(s_r, s_r1, s_r_l[0], s_r_l[1], s_r_l[2], s_r_l[3], s_r_l[4], s_r_l[5], s_r_l[6], s_r_l[7], s_r_l[8], s_r_l[9], s_r_l[10], s_r_l[11], s_r_l[12], s_r_l[13], s_r_l[14], s_r_l[15], s_r_l[16], s_r_l[17], s_r_l[18], s_r_l[19], s_r_l[20], s_r_l[21], s_r_l[22], s_r_l[23], s_r_l[24], s_r_l[25], None, None, None, None)
		elif num_recomm == 27:
			return(s_r, s_r1, s_r_l[0], s_r_l[1], s_r_l[2], s_r_l[3], s_r_l[4], s_r_l[5], s_r_l[6], s_r_l[7], s_r_l[8], s_r_l[9], s_r_l[10], s_r_l[11], s_r_l[12], s_r_l[13], s_r_l[14], s_r_l[15], s_r_l[16], s_r_l[17], s_r_l[18], s_r_l[19], s_r_l[20], s_r_l[21], s_r_l[22], s_r_l[23], s_r_l[24], s_r_l[25], s_r_l[26], None, None, None)
		elif num_recomm == 28:
			return(s_r, s_r1, s_r_l[0], s_r_l[1], s_r_l[2], s_r_l[3], s_r_l[4], s_r_l[5], s_r_l[6], s_r_l[7], s_r_l[8], s_r_l[9], s_r_l[10], s_r_l[11], s_r_l[12], s_r_l[13], s_r_l[14], s_r_l[15], s_r_l[16], s_r_l[17], s_r_l[18], s_r_l[19], s_r_l[20], s_r_l[21], s_r_l[22], s_r_l[23], s_r_l[24], s_r_l[25], s_r_l[26], s_r_l[27], None, None)
		elif num_recomm == 29:
			return(s_r, s_r1, s_r_l[0], s_r_l[1], s_r_l[2], s_r_l[3], s_r_l[4], s_r_l[5], s_r_l[6], s_r_l[7], s_r_l[8], s_r_l[9], s_r_l[10], s_r_l[11], s_r_l[12], s_r_l[13], s_r_l[14], s_r_l[15], s_r_l[16], s_r_l[17], s_r_l[18], s_r_l[19], s_r_l[20], s_r_l[21], s_r_l[22], s_r_l[23], s_r_l[24], s_r_l[25], s_r_l[26], s_r_l[27], s_r_l[28], None)
		elif num_recomm == 30:
			return(s_r, s_r1, s_r_l[0], s_r_l[1], s_r_l[2], s_r_l[3], s_r_l[4], s_r_l[5], s_r_l[6], s_r_l[7], s_r_l[8], s_r_l[9], s_r_l[10], s_r_l[11], s_r_l[12], s_r_l[13], s_r_l[14], s_r_l[15], s_r_l[16], s_r_l[17], s_r_l[18], s_r_l[19], s_r_l[20], s_r_l[21], s_r_l[22], s_r_l[23], s_r_l[24], s_r_l[25], s_r_l[26], s_r_l[27], s_r_l[28], s_r_l[29])
		else:
			return(s_r, s_r1, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None)
	else :
		return(invalid())

def invalid():
	return("Invalid Choice")

def all_reload():
	#--------------------------------------------------------------------------------------------------------------------
	#Load
	dir_path = os.path.dirname(os.path.realpath("df_event.csv"))
	df_event = pd.read_csv(dir_path + "\\df_event.csv", index_col='id')
	df_prof = pd.read_csv(dir_path + "\\df_prof.csv", index_col='id')
	df_unmatch = pd.read_csv(dir_path + "\\df_unmatch.csv", index_col='id')
	finalDf = pd.read_csv(dir_path + "\\finalDf.csv", index_col='id')
	e_unique_str = open("e_unique_str.txt").read().splitlines()
	len_list = open("m_pro_len.txt").readlines()
	m_pro_len = int(len_list[0])
	setting = int(len_list[1])

	#Normalize age
	df_unmatch2 = df_unmatch[df_unmatch.age != 0]
	x2 = df_unmatch2[['age']].values.astype(float)
	min_max_scaler2 = preprocessing.MinMaxScaler()
	x_scaled2 = min_max_scaler2.fit_transform(x2)
	del(df_unmatch2['age'])
	df_unmatch2.insert(loc = m_pro_len-3, column='age', value=x_scaled2)

	array1 = df_prof.values
	array1T = array1.T
	array2 = df_event.values
	array2T = array2.T
	multi_dot_array = array1T.dot(array2)

	list_un = list(map(int,list(df_unmatch2.index)))
	return(list_un, df_unmatch2, multi_dot_array, finalDf, e_unique_str)

#--------------------------------------------------------------------------------------------------------------------
#Lets try dash
external_stylesheets = ['https://codepen.io/amyoshino/pen/jzXypZ.css']

app = dash.Dash(__name__, external_stylesheets = external_stylesheets)

ALLOWED_TYPES = (
    "text"
)

dash_colors = {
	'background': '#F9F9F9',
	'text': '#FFFFFF'
}

app.layout = html.Div([
	html.Div([
		html.H1(
			children = 'Liber Eat - Recommender',
			style= {
				'textAlign': 'left',
				#'color': dash_colors['text']
			},
		),
		html.Br(),
		html.Label('Select the Amount of Recommendations'),
	], className = "row"),

	html.Div([
		dcc.Dropdown(
			id = 'num_recom',
			options = [
				{'label': u'One', 'value': '1'},
				{'label': u'Two', 'value': '2'},
				{'label': u'Three', 'value': '3'},
				{'label': u'Four', 'value': '4'},
				{'label': u'Five', 'value': '5'},
				{'label': u'Six', 'value': '6'},
				{'label': u'Seven', 'value': '7'},
				{'label': u'Eight', 'value': '8'},
				{'label': u'Nine', 'value': '9'},
				{'label': u'Ten', 'value': '10'},
				{'label': u'Eleven', 'value': '11'},
				{'label': u'Twelve', 'value': '12'},
				{'label': u'Thirteen', 'value': '13'},
				{'label': u'Fourteen', 'value': '14'},
				{'label': u'Fifteen', 'value': '15'},
				{'label': u'Sixteen', 'value': '16'},
				{'label': u'Seventeen', 'value': '17'},
				{'label': u'Eighteen', 'value': '18'},
				{'label': u'Nineteen', 'value': '19'},
				{'label': u'Twenty', 'value': '20'},
				{'label': u'Twenty one', 'value': '21'},
				{'label': u'Twenty two', 'value': '22'},
				{'label': u'Twenty three', 'value': '23'},
				{'label': u'Twenty four', 'value': '24'},
				{'label': u'Twenty five', 'value': '25'},
				{'label': u'Twenty six', 'value': '26'},
				{'label': u'Twenty seven', 'value': '27'},
				{'label': u'Twenty eight', 'value': '28'},
				{'label': u'Twenty nine', 'value': '29'},
				{'label': u'Thirty', 'value': '30'},
			],
			placeholder = "Number of recommendations",
			value = ' ',
		),
	], className = "four columns"),

	html.Br(),html.Br(),

	html.Div([
		dcc.Input(id = 'username', value = '', type = 'text'),
		html.Button('Submit', id = 'submit_button', n_clicks=0),
	], className = "row"),

	html.Br(),

	html.Div([
		html.Div(id = 'output_div'),
		html.Div(id = 'output_div1'),
		html.Br(),
		html.Div(id = 'output_div2'),
		html.Div(id = 'output_div3'),
		html.Div(id = 'output_div4'),
		html.Div(id = 'output_div5'),
		html.Div(id = 'output_div6'),
		html.Div(id = 'output_div7'),
		html.Div(id = 'output_div8'),
		html.Div(id = 'output_div9'),
		html.Div(id = 'output_div10'),
		html.Div(id = 'output_div11'),
		html.Div(id = 'output_div12'),
		html.Div(id = 'output_div13'),
		html.Div(id = 'output_div14'),
		html.Div(id = 'output_div15'),
		html.Div(id = 'output_div16'),
		html.Div(id = 'output_div17'),
		html.Div(id = 'output_div18'),
		html.Div(id = 'output_div19'),
		html.Div(id = 'output_div20'),
		html.Div(id = 'output_div21'),
		html.Div(id = 'output_div22'),
		html.Div(id = 'output_div23'),
		html.Div(id = 'output_div24'),
		html.Div(id = 'output_div25'),
		html.Div(id = 'output_div26'),
		html.Div(id = 'output_div27'),
		html.Div(id = 'output_div28'),
		html.Div(id = 'output_div29'),
		html.Div(id = 'output_div30'),
		html.Div(id = 'output_div31'),
	], className = "row"),

	html.Br(),

	html.Div([
		dcc.Markdown("""
			###### *Select from these user ids:*
		"""),
	], className = "row"),

	html.Br(),

	html.Div(style = {'backgroundColor': dash_colors['background']}, children = [
		dcc.Markdown(id = 'text_events'),
	], className = "row"),

], className='ten columns offset-by-one')

@app.callback(
	[Output('text_events', 'children'), Output('output_div', 'children'), Output('output_div1', 'children'), Output('output_div2', 'children'), Output('output_div3', 'children'), Output('output_div4', 'children'), Output('output_div5', 'children'), 
	Output('output_div6', 'children'), Output('output_div7', 'children'), Output('output_div8', 'children'), Output('output_div9', 'children'), Output('output_div10', 'children'), Output('output_div11', 'children'), Output('output_div12', 'children'), 
	Output('output_div13', 'children'), Output('output_div14', 'children'), Output('output_div15', 'children'), Output('output_div16', 'children'), Output('output_div17', 'children'), Output('output_div18', 'children'), Output('output_div19', 'children'), 
	Output('output_div20', 'children'), Output('output_div21', 'children'), Output('output_div22', 'children'), Output('output_div23', 'children'), Output('output_div24', 'children'), Output('output_div25', 'children'), Output('output_div26', 'children'), 
	Output('output_div27', 'children'), Output('output_div28', 'children'), Output('output_div29', 'children'), Output('output_div30', 'children'), Output('output_div31', 'children'),],
	[Input('submit_button', 'n_clicks'), Input('num_recom', 'value'), Input('username', 'value'),])
	#[State('username', 'value')],)
def update_output(btn1, num_recom, username):
	text_un, df_unmatch2, multi_dot_array, finalDf, e_unique_str = all_reload()
	#if n_clicks is not 0:
	changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
	if 'submit_button' in changed_id:
		try:
			sc_input = int(username)
			rec_input = int(num_recom)
			str_r, str_r1, str_r2, str_r3, str_r4, str_r5, str_r6, str_r7, str_r8, str_r9, str_r10, str_r11, str_r12, str_r13, str_r14, str_r15, str_r16, str_r17, str_r18, str_r19, str_r20, str_r21, str_r22, str_r23, str_r24, str_r25, str_r26, str_r27, str_r28, str_r29, str_r30, str_r31 = recomm(sc_input, text_un, df_unmatch2, multi_dot_array, finalDf, e_unique_str, rec_input)
			if rec_input == 1:
				return [text_un, str_r, str_r1, str_r2, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]
			elif rec_input == 2:
				return [text_un, str_r, str_r1, str_r2, str_r3, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]
			elif rec_input == 3:
				return [text_un, str_r, str_r1, str_r2, str_r3, str_r4, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]
			elif rec_input == 4:
				return [text_un, str_r, str_r1, str_r2, str_r3, str_r4, str_r5, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]
			elif rec_input == 5:
				return [text_un, str_r, str_r1, str_r2, str_r3, str_r4, str_r5, str_r6, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]
			elif rec_input == 6:
				return [text_un, str_r, str_r1, str_r2, str_r3, str_r4, str_r5, str_r6, str_r7, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]
			elif rec_input == 7:
				return [text_un, str_r, str_r1, str_r2, str_r3, str_r4, str_r5, str_r6, str_r7, str_r8, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]
			elif rec_input == 8:
				return [text_un, str_r, str_r1, str_r2, str_r3, str_r4, str_r5, str_r6, str_r7, str_r8, str_r9, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]
			elif rec_input == 9:
				return [text_un, str_r, str_r1, str_r2, str_r3, str_r4, str_r5, str_r6, str_r7, str_r8, str_r9, str_r10, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]
			elif rec_input == 10:
				return [text_un, str_r, str_r1, str_r2, str_r3, str_r4, str_r5, str_r6, str_r7, str_r8, str_r9, str_r10, str_r11, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]
			elif rec_input == 11:
				return [text_un, str_r, str_r1, str_r2, str_r3, str_r4, str_r5, str_r6, str_r7, str_r8, str_r9, str_r10, str_r11, str_r12, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]
			elif rec_input == 12:
				return [text_un, str_r, str_r1, str_r2, str_r3, str_r4, str_r5, str_r6, str_r7, str_r8, str_r9, str_r10, str_r11, str_r12, str_r13, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]
			elif rec_input == 13:
				return [text_un, str_r, str_r1, str_r2, str_r3, str_r4, str_r5, str_r6, str_r7, str_r8, str_r9, str_r10, str_r11, str_r12, str_r13, str_r14, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]
			elif rec_input == 14:
				return [text_un, str_r, str_r1, str_r2, str_r3, str_r4, str_r5, str_r6, str_r7, str_r8, str_r9, str_r10, str_r11, str_r12, str_r13, str_r14, str_r15, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]
			elif rec_input == 15:
				return [text_un, str_r, str_r1, str_r2, str_r3, str_r4, str_r5, str_r6, str_r7, str_r8, str_r9, str_r10, str_r11, str_r12, str_r13, str_r14, str_r15, str_r16, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]
			elif rec_input == 16:
				return [text_un, str_r, str_r1, str_r2, str_r3, str_r4, str_r5, str_r6, str_r7, str_r8, str_r9, str_r10, str_r11, str_r12, str_r13, str_r14, str_r15, str_r16, str_r17, None, None, None, None, None, None, None, None, None, None, None, None, None, None]
			elif rec_input == 17:
				return [text_un, str_r, str_r1, str_r2, str_r3, str_r4, str_r5, str_r6, str_r7, str_r8, str_r9, str_r10, str_r11, str_r12, str_r13, str_r14, str_r15, str_r16, str_r17, str_r18, None, None, None, None, None, None, None, None, None, None, None, None, None]
			elif rec_input == 18:
				return [text_un, str_r, str_r1, str_r2, str_r3, str_r4, str_r5, str_r6, str_r7, str_r8, str_r9, str_r10, str_r11, str_r12, str_r13, str_r14, str_r15, str_r16, str_r17, str_r18, str_r19, None, None, None, None, None, None, None, None, None, None, None, None]
			elif rec_input == 19:
				return [text_un, str_r, str_r1, str_r2, str_r3, str_r4, str_r5, str_r6, str_r7, str_r8, str_r9, str_r10, str_r11, str_r12, str_r13, str_r14, str_r15, str_r16, str_r17, str_r18, str_r19, str_r20, None, None, None, None, None, None, None, None, None, None, None]
			elif rec_input == 20:
				return [text_un, str_r, str_r1, str_r2, str_r3, str_r4, str_r5, str_r6, str_r7, str_r8, str_r9, str_r10, str_r11, str_r12, str_r13, str_r14, str_r15, str_r16, str_r17, str_r18, str_r19, str_r20, str_r21, None, None, None, None, None, None, None, None, None, None]
			elif rec_input == 21:
				return [text_un, str_r, str_r1, str_r2, str_r3, str_r4, str_r5, str_r6, str_r7, str_r8, str_r9, str_r10, str_r11, str_r12, str_r13, str_r14, str_r15, str_r16, str_r17, str_r18, str_r19, str_r20, str_r21, str_r22, None, None, None, None, None, None, None, None, None]
			elif rec_input == 22:
				return [text_un, str_r, str_r1, str_r2, str_r3, str_r4, str_r5, str_r6, str_r7, str_r8, str_r9, str_r10, str_r11, str_r12, str_r13, str_r14, str_r15, str_r16, str_r17, str_r18, str_r19, str_r20, str_r21, str_r22, str_r23, None, None, None, None, None, None, None, None]
			elif rec_input == 23:
				return [text_un, str_r, str_r1, str_r2, str_r3, str_r4, str_r5, str_r6, str_r7, str_r8, str_r9, str_r10, str_r11, str_r12, str_r13, str_r14, str_r15, str_r16, str_r17, str_r18, str_r19, str_r20, str_r21, str_r22, str_r23, str_r24, None, None, None, None, None, None, None]
			elif rec_input == 24:
				return [text_un, str_r, str_r1, str_r2, str_r3, str_r4, str_r5, str_r6, str_r7, str_r8, str_r9, str_r10, str_r11, str_r12, str_r13, str_r14, str_r15, str_r16, str_r17, str_r18, str_r19, str_r20, str_r21, str_r22, str_r23, str_r24, str_r25, None, None, None, None, None, None]
			elif rec_input == 25:
				return [text_un, str_r, str_r1, str_r2, str_r3, str_r4, str_r5, str_r6, str_r7, str_r8, str_r9, str_r10, str_r11, str_r12, str_r13, str_r14, str_r15, str_r16, str_r17, str_r18, str_r19, str_r20, str_r21, str_r22, str_r23, str_r24, str_r25, str_r26, None, None, None, None, None]
			elif rec_input == 26:
				return [text_un, str_r, str_r1, str_r2, str_r3, str_r4, str_r5, str_r6, str_r7, str_r8, str_r9, str_r10, str_r11, str_r12, str_r13, str_r14, str_r15, str_r16, str_r17, str_r18, str_r19, str_r20, str_r21, str_r22, str_r23, str_r24, str_r25, str_r26, str_r27, None, None, None, None]
			elif rec_input == 27:
				return [text_un, str_r, str_r1, str_r2, str_r3, str_r4, str_r5, str_r6, str_r7, str_r8, str_r9, str_r10, str_r11, str_r12, str_r13, str_r14, str_r15, str_r16, str_r17, str_r18, str_r19, str_r20, str_r21, str_r22, str_r23, str_r24, str_r25, str_r26, str_r27, str_r28, None, None, None]
			elif rec_input == 28:
				return [text_un, str_r, str_r1, str_r2, str_r3, str_r4, str_r5, str_r6, str_r7, str_r8, str_r9, str_r10, str_r11, str_r12, str_r13, str_r14, str_r15, str_r16, str_r17, str_r18, str_r19, str_r20, str_r21, str_r22, str_r23, str_r24, str_r25, str_r26, str_r27, str_r28, str_r29, None, None]
			elif rec_input == 29:
				return [text_un, str_r, str_r1, str_r2, str_r3, str_r4, str_r5, str_r6, str_r7, str_r8, str_r9, str_r10, str_r11, str_r12, str_r13, str_r14, str_r15, str_r16, str_r17, str_r18, str_r19, str_r20, str_r21, str_r22, str_r23, str_r24, str_r25, str_r26, str_r27, str_r28, str_r29, str_r30, None]
			elif rec_input == 30:
				return [text_un, str_r, str_r1, str_r2, str_r3, str_r4, str_r5, str_r6, str_r7, str_r8, str_r9, str_r10, str_r11, str_r12, str_r13, str_r14, str_r15, str_r16, str_r17, str_r18, str_r19, str_r20, str_r21, str_r22, str_r23, str_r24, str_r25, str_r26, str_r27, str_r28, str_r29, str_r30, str_r31]
		except ValueError:
			return [text_un, invalid(), None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]
	return [text_un, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]

if __name__ == '__main__':
	app.run_server(port = 6060)