# import all required libraries.
import matplotlib
from flask import Flask, render_template,request
import seaborn as sns
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import turicreate as tc
matplotlib.use('Agg')




IMAGES_FOLDER = os.path.join('static', 'images')

app = Flask(__name__)

sns.set_style('white')
sns.set_style('ticks')

app.config['UPLOAD_FOLDER'] = IMAGES_FOLDER
df = pd.read_csv("NFHS-4_NFHS3_Factsheet-All_India_Indicators_R1.csv")
model = tc.load_model('multiple_lr_model.tc') #load Multiple Regression Model

features_names = ['first_trimester_check', #features list for model
 'at_least_4_checks',
 'tetanus_vaccination_mothers',
 'folic_acid_consumed',
 'full_care',
 'MCP_card',
 'postnatal_care',
 'financial_assistance',
 'avg_expenditure',
 'home_post_partum_check',
 'check_2_days',
 'institutional_births']



@app.route("/index.html")
def home():
     return render_template("index.html")

@app.route("/")
@app.route('/index')
def histogram(): #generate histogram and scatter plots
    
    for i in ['Population and Household Profile - Population below age 15 years (%)',
              'Population and Household Profile - Population (female) age 6 years and above who ever attended school (%)',
              'Marriage and Fertility - Women age 15-19 years who were already mothers or pregnant at the time of the survey (%)',
              'Delivery Care (for births in the 5 years before the survey) - Institutional births in public facility (%)',
              'Maternity Care (for last birth in the 5 years before the survey) - Children born at home who were taken to a health facility for check-up within 24 hours of birth (%)',

              'Child Feeding Practices and Nutritional Status of Children - Children under 5 years who are underweight (weight-for-age) (%)',
              ]:
        full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'fig' + i + '.png')
        df.hist(column=i)
        plt.savefig(full_filename)
    x_list = ['Population and Household Profile - Children under age 5 years whose birth was registered (%)',
                  'Marriage and Fertility - Women age 15-19 years who were already mothers or pregnant at the time of the survey (%)',
                  'Maternity Care (for last birth in the 5 years before the survey) - Children born at home who were taken to a health facility for check-up within 24 hours of birth (%)',
                  'Maternity Care (for last birth in the 5 years before the survey) - Children born at home who were taken to a health facility for check-up within 24 hours of birth (%)',
                  'Child Feeding Practices and Nutritional Status of Children - Breastfeeding children age 6-23 months receiving an adequate diet (%',
                  'Child Feeding Practices and Nutritional Status of Children - Non-breastfeeding children age 6-23 months receiving an adequate diet (%',
                  'Child Feeding Practices and Nutritional Status of Children - Children under 5 years who are wasted (weight-for-height) (%)',
                  'Treatment of Childhood Diseases (children under age 5 years) - Prevalence of symptoms of acute respiratory infection (ARI) in the last 2 weeks preceding the survey (%)',
                  ]

    y_list = ['Infant and Child Mortality Rates (per 1000 live births) - Under-five mortality rate (U5MR)',
                  'Infant and Child Mortality Rates (per 1000 live births) - Infant mortality rate (IMR)',
                  'Delivery Care (for births in the 5 years before the survey) - Institutional births in public facility (%)',
                  'Delivery Care (for births in the 5 years before the survey) - Home delivery conducted by skilled health personnel (out of total deliveries) (%)',
                  'Infant and Child Mortality Rates (per 1000 live births) - Under-five mortality rate (U5MR)',
                  'Infant and Child Mortality Rates (per 1000 live births) - Under-five mortality rate (U5MR)',
                  'Infant and Child Mortality Rates (per 1000 live births) - Under-five mortality rate (U5MR)',
                  'Infant and Child Mortality Rates (per 1000 live births) - Under-five mortality rate (U5MR)',
                  ]

    x_labels = ['Children under age 5 years whose birth was registered (%)',
                    'Women age 15-19 years who were already mothers or pregnant at the time of the survey (%)',
                    'Children born at home who were taken to a hospitals within 24 hours of birth (%)',
                    'Children born at home who were taken to a hospitals within 24 hours of birth (%)',
                    'Breastfeeding children age 6-23 months receiving an adequate diet (%)',
                    'Non-breastfeeding children age 6-23 months receiving an adequate diet (%)',
                    'Children under 5 years who are wasted (weight-for-height) (%)',
                    'Prevalence of symptoms of acute respiratory infection (ARI) in the last 2 weeks preceding the survey (%)',
                    ]

    y_labels = ['Under-five mortality rate (U5MR)',
                    'Infant mortality rate (IMR)',
                    'Infant mortality rate (IMR)1',
                    'Home delivery conducted by skilled health personnel (out of total deliveries) (%)',
                    'Under-five mortality rate (U5MR)1',
                    'Under-five mortality rate (U5MR)2',
                    'Under-five mortality rate (U5MR)3',
                    'Under-five mortality rate (U5MR)4'
               ]

    full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'figsca1.png')
    fig = sns.regplot(
        x='Marriage and Fertility - Women age 15-19 years who were already mothers or pregnant at the time of the survey (%)',
        y='Infant and Child Mortality Rates (per 1000 live births) - Infant mortality rate (IMR)', color='green',
        data=df)

    plt.xlabel('Women age 15-19 years who were already mothers or pregnant at the time of the survey (%)')
    plt.ylabel('Infant mortality rate (IMR)')

    plt.savefig(full_filename)

    full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'figsca3.png')
    fig = sns.regplot(
        x='Maternity Care (for last birth in the 5 years before the survey) - Children born at home who were taken to a health facility for check-up within 24 hours of birth (%)',
        y='Delivery Care (for births in the 5 years before the survey) - Institutional births in public facility (%)',
        color='yellow', data=df)
    plt.xlabel('Children born at home who were taken to a hospitals within 24 hours of birth (%)')
    plt.ylabel('Infant mortality rate (IMR)')

    plt.savefig(full_filename)

    full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'figsca4.png')
    fig = sns.regplot(
        x='Maternity Care (for last birth in the 5 years before the survey) - Children born at home who were taken to a health facility for check-up within 24 hours of birth (%)',
        y='Delivery Care (for births in the 5 years before the survey) - Home delivery conducted by skilled health personnel (out of total deliveries) (%)',
        color='purple', data=df)
    plt.xlabel('Children born at home who were taken to a hospitals within 24 hours of birth (%)')
    plt.ylabel('Home delivery conducted by skilled health personnel (out of total deliveries) (%)')

    plt.savefig(full_filename)

    full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'figsca5.png')
    fig = sns.regplot(
        x='Child Feeding Practices and Nutritional Status of Children - Breastfeeding children age 6-23 months receiving an adequate diet (%',
        y='Infant and Child Mortality Rates (per 1000 live births) - Under-five mortality rate (U5MR)',
        color='violet', data=df)
    plt.xlabel('Breastfeeding children age 6-23 months receiving an adequate diet (%)')
    plt.ylabel('Under-five mortality rate (U5MR)')

    plt.savefig(full_filename)


    full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'figsca6.png')
    sns.regplot(
        x='Child Feeding Practices and Nutritional Status of Children - Children under 5 years who are wasted (weight-for-height) (%)',
        y='Infant and Child Mortality Rates (per 1000 live births) - Under-five mortality rate (U5MR)',
        color='brown', data=df)
    plt.xlabel('Children under 5 years who are wasted (weight-for-height) (%)')
    plt.ylabel('Infant and Child Mortality Rates (per 1000 live births) - Under-five mortality rate (U5MR')
    plt.savefig(full_filename)

    full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'figsca7.png')
    sns.regplot(
        x='Treatment of Childhood Diseases (children under age 5 years) - Prevalence of symptoms of acute respiratory infection (ARI) in the last 2 weeks preceding the survey (%)',
        y='Infant and Child Mortality Rates (per 1000 live births) - Under-five mortality rate (U5MR)',
        color='purple', data=df)

    plt.xlabel(
        'Prevalence of symptoms of acute respiratory infection (ARI) in the last 2 weeks preceding the survey (%)')
    plt.ylabel(' Under-five mortality rate (U5MR)')

    plt.savefig(full_filename)




    return render_template("index.html",
                           chart1=os.path.join(app.config['UPLOAD_FOLDER'], 'figPopulation and Household Profile - Population below age 15 years (%).png'),
                           chart2=os.path.join(app.config['UPLOAD_FOLDER'], 'figPopulation and Household Profile - Population (female) age 6 years and above who ever attended school (%).png'),
                           chart3=os.path.join(app.config['UPLOAD_FOLDER'], 'figMarriage and Fertility - Women age 15-19 years who were already mothers or pregnant at the time of the survey (%).png'),
                           chart4=os.path.join(app.config['UPLOAD_FOLDER'], 'figDelivery Care (for births in the 5 years before the survey) - Institutional births in public facility (%).png'),
                           chart5=os.path.join(app.config['UPLOAD_FOLDER'], 'figMaternity Care (for last birth in the 5 years before the survey) - Children born at home who were taken to a health facility for check-up within 24 hours of birth (%).png'),

                           chart6=os.path.join(app.config['UPLOAD_FOLDER'], 'figChild Feeding Practices and Nutritional Status of Children - Children under 5 years who are underweight (weight-for-age) (%).png'),


                           chart7=os.path.join(app.config['UPLOAD_FOLDER'], 'figca1.png'),


                           chart9=os.path.join(app.config['UPLOAD_FOLDER'], 'si.png'),
                           chart10=os.path.join(app.config['UPLOAD_FOLDER'], 's4.png'),
                           chart11=os.path.join(app.config['UPLOAD_FOLDER'], 's5.png'),
                           chart12=os.path.join(app.config['UPLOAD_FOLDER'], 's6.png'),
                           chart13=os.path.join(app.config['UPLOAD_FOLDER'], 's7.png'),



     )



#prediction method
@app.route('/predict',methods=['POST','GET'])
def predict():
    print('\n\nrequest.form = ')
    print(request.form)
    input_values = list(request.form.values())
    input_values = input_values[:-1]
    ('\n\nfeatures values = ')
    print(input_values)

    query_input={}
    for i in range(12):
        if input_values[i] != '':        
            query_input[features_names[i]] = float(input_values[i])
        else:
            query_input[features_names[i]] = 0
    query_output = model.predict(query_input)
    print(query_output)
    query_output = round(query_output[0], 2)
    
#    return render_template('index.html', pred='Test Return Statement')

    if query_output>50:
        return render_template('index.html',pred='Area is under critical condition.\n Estimated IMR is {}'.format(query_output))
    else:
        return render_template('index.html',pred='Normal condition.\n Estimated IMR is {}'.format(query_output))


@app.route('/goa.html')
def goa():
    return render_template('goa.html')



@app.route('/anp.html')
def anp():
    return render_template('anp.html')


@app.route('/ap.html')
def ap():
    return render_template('ap.html')

@app.route('/assam.html')
def assam():
    return render_template('assam.html')


@app.route('/bihar.html')
def bihar():
    return render_template('bihar.html')

@app.route('/chattisgarh.html')
def chattisgarh():
    return render_template('chattisgarh.html')

@app.route('/gujarat.html')
def gujrat():
    return render_template('gujarat.html')

@app.route('/haryana.html')
def haryana():
    return render_template('haryana.html')

@app.route('/hp.html')
def hp():
    return render_template('hp.html')

@app.route('/jandk.html')
def jandk():
    return render_template('jandk.html')

@app.route('/jharkhand.html')
def jharkhand():
    return render_template('jharkhand.html')

@app.route('/karnataka.html')
def karnataka():
    return render_template('karnataka.html')

@app.route('/kerala.html')
def kerala():
    return render_template('kerala.html')

@app.route('/maharashtra.html')
def maharashtra():
    return render_template('maharashtra.html')

@app.route('/manipur.html')
def manipur():
    return render_template('manipur.html')

@app.route('/meghalaya.html')
def meghalaya():
    return render_template('meghalaya.html')

@app.route('/mizoram.html')
def mizoram():
    return render_template('mizoram.html')

@app.route('/mp.html')
def mp():
    return render_template('mp.html')

@app.route('/nagaland.html')
def nagaland():
    return render_template('nagaland.html')

@app.route('/odissha.html')
def odissha():
    return render_template('odissha.html')

@app.route('/punjab.html')
def punjab():
    return render_template('punjab.html')

@app.route('/rajasthan.html')
def rajasthan():
    return render_template('rajasthan.html')

@app.route('/sikkim.html')
def sikkim():
    return render_template('sikkim.html')

@app.route('/tamilnadu.html')
def tamilnadu():
    return render_template('tamilnadu.html')

@app.route('/telangana.html')
def telangana():
    return render_template('telangana.html')

@app.route('/tripura.html')
def tripura():
    return render_template('tripura.html')

@app.route('/up.html')
def up():
    return render_template('up.html')

@app.route('/uttarakhand.html')
def uttarakhand():
    return render_template('uttarakhand.html')

@app.route('/wp.html')
def wp():
    return render_template('wp.html')

@app.route('/features.html')
def features():
    return render_template('features.html')

@app.route('/docs.html')
def docs():
    return render_template('docs.html')





if __name__ == "__main__":
    app.run(debug=True)


