# pill_dictionary.py

class PillDictionary:
    def __init__(self):
        self.pill_info = {
            'ACECLOFENAC, PARACETAMOL & SERRATIOPEPTIDASE': {
                'benefits': 'Aceclofenac, Paracetamol, and Serratiopeptidase offer pain relief and anti-inflammatory benefits. Aceclofenac reduces pain and inflammation, Paracetamol provides mild to moderate pain relief and reduces fever, while Serratiopeptidase helps alleviate inflammation and may aid in tissue healing.',
                'tablets': ['Aceclospa', 'Paraserr', 'Zerodol-SP', 'Hifenac-P', 'Aceclo-SR', 'Dolowin Plus', 'Ace-Proxyvon SP', 'Arflur Plus', 'Trinicalm Plu', 'Diclomol Plus'],
                'side_effects': 'Common side effects include stomach pain, loss of appetite, indigestion, stomach upset, nausea, diarrhea, dizziness, and increased liver enzymes.',
                'dosage': 'Adult : 500 - 1000 mg in 3 times daily , Child : 60 mg / kg body weight /day in 4 divided doses.'
            },
            'AMOXYCILLIN & POTASSIUM CLAVULANATE': {
                'benefits': 'Amoxycillin & Potassium Clavulanate is used to treat certain infections caused by bacteria, including infections of the ears, lungs, sinus, skin, and urinary tract.',
                'tablets': ['Augmentin', 'Clavam', 'Moxikind-CV ', 'Megamentin', 'Claventin','Clavamox', 'Synulox ','Moxclav ', 'Clavix', 'Amoxyclav'],
                'side_effects': 'Common side effects include Rash, Diarrhea, Fever, Head ache, Easy bruising or bleeding, Itching, Nausea, Swelling of the face, Allergic. ',
                'dosage': 'Adult : Adults and children weighing 40 kilograms (kg) or more—250 milligrams (mg) amoxicillin and 125 mg clavulanate every 8 hours or 500 mg amoxicillin and 125 mg clavulanate every 12 hours , Child : Children weighing less than 40 kg—Use and dose must be determined by your doctor.'
            },
            'BETAHISTINE HYDROCHLORIDE': {
                'benefits': "Betahistine hydrochloride benefits include reducing vertigo symptoms, improving vestibular function, and managing Meniere's disease with minimal side effects.",
                'tablets': ['Vertin', 'Betavert ', 'Vertin OD', 'Vertin OD', 'Betaserc', 'Vestin', 'Cizerta', 'Zetast', 'Betaday ', 'Serc'],
                'side_effects': 'Common side effects of Betahistine hydrochloride may include mild gastrointestinal symptoms such as nausea or stomach upset.',
                'dosage': 'Adult : Adults is usually 16 to 48 mg taken orally in divided doses throughout the day , Child : Use and dose must be determined by your doctor.'
            },
            'DICYCLOMINE HYDROCHLORIDE & PARACETAMOL': {
                'benefits': 'Dicyclomine Hydrochloride Paracetamol is used in the treatment of abdominal pain. It relieves abdominal pain that is associated with biliary colic, intestinal colic, renal colic and dysmenorrhea.',
                'tablets': ['Spasmonil Plus', 'Cyclopam Plus', 'Colimex-Plus', 'Spasmo Proxyvon Plus', 'Spasmo-Plus', 'Drotin Plus', 'Spasmol D',
                            'Colirid-P', 'Meftal-Spas', 'Spasmo-Relax'],
                'side_effects': 'Common side effects of Dicyclomine Hydrochloride Paracetamol  may include Nausea,  Xerostomia, Dizziness, Sleepiness, Weakness, Blurred Vision , Confusion',
                'dosage': 'Adult : The usual oral dose for adults is 10-20 mg, taken 3-4 times daily , Child : Use and dose must be determined by your doctor.'
            },
            'INDOMETHACIN': {
                'benefits': 'Indomethacin provides effective pain relief, reduces inflammation, lowers fever, and is used for conditions like arthritis, gout, and closure of patent ductus arteriosus (PDA).',
                'tablets': ['Indocid', 'Indocap ', 'Indomethacin', 'Indocap SR', 'Indo-P', 'Indocap SR', 'Indosin',
                            'Indonil', 'Indo Top', 'Indolan'],
                'side_effects': 'Common side effects of indomethacin may include Stomach upset or pain, Nausea, Heartburn, Diarrhea, Headache, Dizziness, Drowsiness, Fluid retention, High blood pressure, Increased risk of bleeding',
                'dosage': 'Adult :  The usual starting dose for adults is 25 mg two or three times daily , Child :  the starting dose is 1-2 mg/kg per day, divided into multiple doses..'
            },
            'ITRACONAZOLE': {
                'benefits': 'Itraconazole is an antifungal medication effective against a wide range of fungal infections, including systemic, oral, and nail infections. It offers convenient oral administration, minimal side effects, and long-lasting action, making it a valuable treatment option for patients.',
                'tablets': ['Itrasys', 'Canditral', 'Itrazol', 'Candiforce', 'Sporanox', 'Itzhh', 'Candistat',
                            'Onitraz', 'Zocon', 'Sporanox LS'],
                'side_effects': 'Common side effects include Decreased urine output.dry mouth.increased thirst.irregular heartbeat.muscle pain or cramps.nausea.numbness or tingling in the hands, feet, or lips.',
                'dosage': 'Adult :  200 mg once daily , Child : Use and dose must be determined by your doctor.'
            },


            'MONTELUKAST SODIUM & LEVOCETIRIZINE HYDROCHLORIDE': {
                'benefits': 'Montelukast sodium and levocetirizine hydrochloride, used together, provide relief from allergic rhinitis and asthma symptoms. Montelukast blocks inflammation in the airways, while levocetirizine blocks histamine, reducing symptoms like nasal congestion, sneezing, and itching.',
                'tablets': ['Montair-LC', 'Levocet-M', 'Montecip-LC', 'Montek LC', 'Lukotas-LC', 'Monticope', 'Montair Plus', 'Xyzal-M ', 'Livocet-M', 'Singulair-M'],
                'side_effects': 'Common side effects include Nausea, Skin rash, Diarrhoea, Vomiting, Dry mouth, Headache, Skin rash, Fatigue (Weakness)',
                'dosage': 'Adult : 10 mg montelukast and 5 mg levocetirizine , Child : 4 mg montelukast and 2.5 mg levocetirizine .'
            },
            'PANTOPRAZOLE GASTRO-RESISTANT': {
                'benefits': 'Pantoprazole gastro-resistant tablets help alleviate symptoms of acid-related disorders such as gastroesophageal reflux disease (GERD), peptic ulcers, and gastritis by reducing stomach acid production.',
                'tablets': ['Pantocid', 'Pan', 'Pantop', 'Pantocid-D', 'PPI-20', 'Pantocid-DSR', 'Pantocid-IT ',
                            'Pan-D ', 'Pantop-D ', 'Pan-L'],
                'side_effects': 'Common side effects of Pantoprazole gastro-resistant tablets may include Headache Nausea Diarrhea Abdominal pain Constipation Flatulence Dizziness Rash or itching Fatigue Joint pain',
                'dosage': 'Adult :  The usual dose for adults is 20-40 mg once daily, taken before a meal  , Child : Use and dose must be determined by your doctor.'
            },
            'PARACETAMOL': {
                'benefits': 'Paracetamol provides effective relief from mild to moderate pain and fever, with minimal side effects and wide availability over-the-counter.',
                'tablets': ['Crocin', 'Calpol', 'Dolo-650', 'Metacin', 'Pacimol', 'Pyrigesic', 'Sumo', 'Paracip', 'Zerodol-P', 'Anacin'],
                'side_effects': 'Common side effects include tiredness, breathlessness, your fingers and lips to go blue, anaemia (low red blood cell count), liver and kidney damage, heart disease and stroke if you have high blood pressure.',
                'dosage': 'Adult : 500 mg in 3 times daily , Child : 60 mg / kg body weight /day in 4 divided doses.',
                'drug class':'abc.'
            }
        }

    def get_pill_info(self, pill_name):
        return self.pill_info.get(pill_name, {'benefits': 'Not available',  'tablets': [],
                                              'side_effects': 'Not available', 'dosage': 'Not available', 'drug class':'not available'})