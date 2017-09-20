import sys
import pickle
from sklearn.feature_extraction import DictVectorizer
import numpy as np
from sklearn.linear_model import LogisticRegression

head_list = ["head", "top", "vice", "senior", "prime", "junior", "general", "chief", "health",
                 "police", "environment", "security", "education", "defensive", "economy", "finance"]

abravation = ["dr.","fr.","esq.","hon.","jr.","mr.","mrs.","ms.","messrs.","mmes.","msgr.","prof.","aprof","rev.","rt.","hon.","sr.","st."]

occupation_list = ['-','accommodation','warden','accounting','technician','accounts','clerk','-','bookkeeper','acoustics','consultant',
                   'actor','actuary','acupuncturist','administrative','assistant','adult','nurse','advertising','account','executive',
                   'advertising','account','planner','advertising','art','director','advertising','copywriter','advertising','media',
                   'buyer','advertising','media','planner','aerospace','engineer','aerospace','engineering','technician','agricultural',
                   'contractor','agricultural','engineer','agricultural','engineering','technician','agricultural','inspector','air','cabin',
                   'crew','air','traffic','controller','airline','customer','service','agent','airline','pilot','airport','baggage','handler',
                   'airport','information','assistant','alexander','technique','teacher','ambulance','care','assistant','ambulance','paramedic',
                   'anaesthetist','analytical','textile','technologist','anatomical','pathology','technician','animal','care','worker',
                   'animal','technician','animator','antique','dealer','arboricultural','officer','archaeologist','architect','architectural',
                   'technician','or','technologist','archivist','army','officer','army','soldier','aromatherapist','art','editor','art',
                   'gallery','curator','art','therapist','art','valuer','arts','administrator','assistance','dog','trainer','assistant',
                   'immigration','officer','astronaut','astronomer','audiologist','audio-visual','technician','auditor','auto','electrician',
                   'automotive','engineer','bailiff','baker','bank','manager','banking','customer','service','adviser','bar','person',
                   'barista','barrister','barristers','clerk','beauty','consultant','beauty','therapist','beekeeper','betting','shop',
                   'cashier','bid','writer','bilingual','secretary','biochemist','biologist','biomedical','scientist','biotechnologist',
                   'blacksmith','body','piercer','bodyguard','bomb','disposal','technician','bookbinder','or','print','finisher',
                   'bookmaker','bookseller','border','force','officer','or','assistant','officer','botanist','bottler','brewery',
                   'worker','bricklayer','british','sign','language','interpreter','broadcast','engineer','broadcast','journalist',
                   'builders','merchant','building','control','officer','building','services','engineer','building','surveyor','building',
                   'technician','bus','or','coach','driver','business','adviser','business','analyst','business','development','manager',
                   'butcher','butler','cabinet','maker','cad','technician','cake','decorator','call','centre','operator','car','fleet',
                   'manager','car','rental','agent','car','salesperson','car','valet','care','assistant','care','home','advocate',
                   'careers','adviser','caretaker','carpenter','or','joiner','carpet','fitter-floor','layer','cartographer','catering',
                   'manager','cavity','insulation','installer','ceiling','fixer','cellar','technician','cemetery','worker','ceramics',
                   'designer-maker','charity','fundraiser','chauffeur','checkout','operator','chef','chemical','engineer','chemical',
                   'engineering','technician','chemical','plant','process','operator','chemist','childminder','nurse','chimney','sweep',
                   'chiropodist' ,'podiatrist','chiropody' ,'podiatry','assistant','chiropractor','choreographer','cinema','projectionist',
                   'cinema-theatre','attendant','cinematographer','circus','performer','civil','enforcement','officer','civil','engineer',
                   'civil','engineering','technician','civil','service','administrative','officer','civil','service','executive','officer',
                   'classical','musician','cleaner','clerk','of','works','clinical','engineer','clinical','psychologist','clinical',
                   'scientist','clothing','alteration','hand','clothing','presser','cnc','machinist','coastguard','cognitive','behavioural',
                   'therapist','colon','hydrotherapist','colour','therapist','commercial','energy','assessor','commissioning','editor',
                   'communication','support','worker','community','arts','worker','community','development','worker','community','education',
                   'coordinator','community','matron','community','transport','driver','community','transport','operations','manager',
                   'community','transport','passenger','assistant','company','secretary','computer','games','developer','computer',
                   'games','tester','computer','service','and','repair','technician','conference','and','exhibition','organiser',
                   'conservator','construction','contracts','manager','construction','manager','construction','operative','construction',
                   'plant','mechanic','construction','plant','operator','consumer','scientist','copy','editor','coroner','costume',
                   'designer','counsellor','counter','service','assistant','countryside','officer','countryside','ranger','courier',
                   'court','administrative','officer','court','legal','adviser','court','usher','credit','controller','credit','manager',
                   'crematorium','technician','criminal','intelligence','analyst','critical','care','technologist','croupier','crown',
                   'prosecutor','cruise','ship','steward','customer','service','assistant','customer','services','manager','customs',
                   'officer','cycle','mechanic','cycling','coach','​dance','movement','psychotherapist','dance','teacher','dancer',
                   'data','analyst-statistician','data','entry','clerk','database','administrator','delivery','van','driver',
                   'demolition','operative','dental','hygienist','dental','nurse','dental','technician','dental','therapist','dentist',
                   'design','engineer','dietitian','digital','marketing','officer','diplomatic','service','officer','dispensing',
                   'optician','district','nurse','diver','dj','doctor','-','gp','dog','groomer','dog','handler','domestic','appliance',
                   'service','engineer','domestic','energy','assessor','door','supervisor','dramatherapist','dressmaker','driving',
                   'instructor','drug','and','alcohol','worker','dry','liner','dry-cleaner','early','years','teacher','ecologist',
                   'economic','development','officer','economist','editorial','assistant','education','technician','education','welfare',
                   'officer','efl','teacher','e-learning','developer','electrical','engineer','electrical','engineering','technician',
                   'electrician','electricity','distribution','worker','electricity','generation','worker','electronics','engineer',
                   'electronics','engineering','technician','embalmer','emergency','care','assistant','emergency','medical','dispatcher',
                   'energy','engineer','engineering','construction','craftworker','engineering','construction','technician','engineering',
                   'craft','machinist','engineering','maintenance','fitter','engineering','maintenance','technician','engineering',
                   'operative','entertainer','entertainment','agent','environmental','consultant','environmental','health','officer',
                   'equalities','officer','ergonomist','estate','agent','estates','officer','estimator','european','union','official',
                   'events','manager','exhibition','designer','facilities','manager','fairground','worker','family','mediator','family',
                   'support','worker','farm','manager','farm','secretary','farm','worker','farrier','fashion','design','assistant','fashion',
                   'designer','fashion','model','fence','installer','financial','adviser','financial','services','customer','adviser',
                   'fine','artist','fingerprint','officer','firefighter','fish','farmer','fishing','vessel','deckhand','fishing','vessel',
                   'skipper','fitness','instructor','florist','food','packaging','operative','food','processing','worker','food',
                   'scientist-food','technologist','football','coach','football','referee','football','agent','sports','agent',
                   'footballer','footwear','designer','footwear','manufacturing','operative','forensic','computer','analyst','forensic',
                   'psychologist','forensic','scientist','forest','officer','forest','worker','forklift','truck','engineer','forklift',
                   'truck','operator','foster','carer','foundry','moulder','foundry','patternmaker','foundry','process','operator',
                   'franchise','owner','freight','forwarder','french','polisher','funeral','director','furniture','designer','furniture',
                   'restorer','further','education','lecturer','gamekeeper','gardener','garment','technologist','gas','mains','layer',
                   'gas','service','technician','general','practice','surveyor','geneticist','geoscientist','geotechnician','glass',
                   'engraver','glassmaker','glazier','gp','practice','manager','grants','officer','graphic','designer','groundsperson',
                   'or','greenkeeper','hairdresser','handyperson','hat','designer','or','milliner','headteacher','head','teacher',
                   'health','and','safety','adviser','health','play','specialist','health','promotion','specialist','health','records',
                   'clerk','health','service','manager','health','trainer','health','visitor','healthcare','assistant','healthcare',
                   'science','assistant','heat','treatment','operator','heating','and','ventilation','engineer','helicopter','pilot',
                   'helpdesk','professional','her','majesty','inspector-regulatory','inspector','higher','education','lecturer','highways',
                   'cleaner','homeopath','horse','groom','horse','riding','instructor','horticultural','manager','horticultural',
                   'therapist','horticultural','worker','hospital','doctor','hospital','porter','hotel','manager','hotel','porter',
                   'hotel','receptionist','hotel','room','attendant','housekeeper','housing','officer','housing','policy','officer',
                   'human','resources','officer','hypnotherapist','​​​​​illustrator','image','consultant','immigration','adviser','immigration',
                   'officer','indexer','information','scientist','insurance','account','manager','insurance','broker','insurance','claims',
                   'handler','insurance','claims','manager','insurance','loss','adjuster','insurance','risk','surveyor','insurance',
                   'technician','insurance','underwriter','interior','designer','interpreter','investment','analyst','it','project',
                   'manager','it','security','coordinator','it','support','technician','it','trainer','jewellery','designer-maker',
                   'kennel','worker','kitchen','and','bathroom','fitter','kitchen','assistant','kitchen','manager','head','chef',
                   'kitchen','porter','laboratory','technician','land','and','property','valuer','and','auctioneer','land',
                   'surveyor','landscape','architect','landscaper','large','goods','vehicle','driver','laundry','worker','leakage',
                   'operative','learning','disability','nurse','learning','mentor','leather','craftworker','leather','technologist',
                   'legal','executive','legal','secretary','leisure','centre','assistant','leisure','centre','manager','letting','agent',
                   'librarian','library','assistant','licensed','conveyancer','life','coach','lifeguard','lift','engineer','light',
                   'industry','assembler','lighting','technician','live','sound','engineer','local','government','administrative',
                   'assistant','local','government','officer','local','government','revenues','officer','locksmith','machine','printer',
                   'magazine','journalist','make-up','artist','management','accountant','management','consultant','manufacturing',
                   'supervisor','manufacturing','systems','engineer','marine','craftsperson','marine','engineer','marine','engineering',
                   'technician','market','research','data','analyst','market','research','executive','market','research','interviewer',
                   'market','trader','marketing','executive','marketing','manager','martial','arts','instructor','massage','therapist',
                   'materials','engineer','materials','technician','maternity','support','worker','measurement','and','control','engineer',
                   'measurement','and','control','technician','meat','hygiene','inspector','meat','process','worker','mechanical',
                   'engineer','mechanical','engineering','technician','media','researcher','medical','herbalist','medical','illustrator',
                   'medical','physicist','medical','sales','representative','medical','secretary','member','of','parliament','mental',
                   'health','nurse','merchant','navy','deck','officer','merchant','navy','engineering','officer','merchant','navy',
                   'rating','meteorologist','microbiologist','midwife','minerals','surveyor','mobile','catering','assistant','model',
                   'maker','money','adviser','or','debt','counsellor','montessori','teacher','mortgage','adviser','motor','vehicle',
                   'body','repairer','motor','vehicle','breakdown','engineer','motor','vehicle','fitter','motor','vehicle','parts',
                   'person','motor','vehicle','technician','motorsport','engineer','museum','assistant','museum','curator','music',
                   'promotions','manager','music','teacher','music','therapist','musical','instrument','maker','musical','instrument',
                   'repairer','nail','technician','nanny','nanotechnologist','naturopath','naval','architect','neighbourhood','warden',
                   'network','engineer','network','manager','newspaper','journalist','newspaper','or','magazine','editor',
                   'non-destructive','testing','technician','nuclear','engineer','nurse','nurse','children','nurse','nurse','nurse',
                   'learning','disability','nurse','mental','health','nurse','occupational','health','nurse','nurse','nurse','nursery',
                   'manager','nursery','worker','nutritional','therapist','nutritionist','occupational','health','nurse','occupational',
                   'therapist','occupational','therapy','support','worker','oceanographer','office','equipment','service','technician',
                   'offshore','drilling','worker','offshore','roustabout','online','tutor','operating','department','practitioner',
                   'operational','researcher','optometrist','ornithologist','orthoptist','osteopath','outdoor','activities','instructor',
                   '​​​​packaging','technologist','packer','paint','sprayer','painter','and','decorator','palaeontologist','palliative',
                   'care','assistant','paper','manufacturing','operative','paper','technologist','paralegal','paramedic','patent',
                   'attorney','pathologist','patient','advice','and','liaison','service','officer','patient','transport','service',
                   'controller','pattern','cutter','pattern','grader','payroll','administrator','payroll','manager','pe','teacher',
                   'pensions','administrator','pensions','adviser','pensions','manager','personal','assistant','personal','shopper',
                   'personal','trainer','pest','control','technician','pet','behaviour','counsellor','pet','shop','assistant','petrol',
                   'service','sales','assistant','pharmacist','pharmacologist','pharmacy','technician','phlebotomist','photographer',
                   'photographic','stylist','photographic','technician','physicist','physiotherapist','physiotherapy','assistant',
                   'picture','framer','pilates','teacher','planning','and','development','surveyor','plasterer','plastics','process',
                   'worker','play','therapist','playworker','plumber','podiatrist' ,'chiropodist','podiatry' ,'chiropody','assistant',
                   'police','community','support','officer','police','officer','pop','musician','port','operative','portage','home',
                   'visitor','post','office','customer','service','assistant','postal','delivery','worker','practice','nurse','pre-press',
                   'operator','primary','care','graduate','mental','health','worker','primary','school','teacher','printing','administrator',
                   'prison','governor','prison','instructor','prison','officer','private','investigator','private','practice',
                   'accountant','probation','officer','probation','services','officer','product','designer','production','manager',
                   'production','worker','project','manager','proofreader','prop','maker','prosthetist-orthotist','psychiatrist',
                   'psychologist','psychotherapist','public','finance','accountant','public','relations','officer','publican-licensee',
                   'purchasing','manager','qcf','assessor','quality','control','technician','quality','manager','quantity','surveyor',
                   'quarry','engineer','quarry','operative','radio','broadcast','assistant','radiographer','radiography','assistant',
                   'raf','airman','or','airwoman','raf','non-commissioned','aircrew','raf','officer','rail','engineering','technician',
                   'rail','track','maintenance','worker','receptionist','recruitment','consultant','recycled','metals','worker','recycling',
                   'officer','recycling','operative','reflexologist','refrigeration','and','air','conditioning','engineer','refuse',
                   'collector','registered','care','home','manager','registrar','of','births,','deaths,','marriages','and','civil',
                   'partnerships','reiki','healer','religious','leader','removals','worker','reprographic','assistant','research',
                   'scientist','residential','support','worker','resort','representative','restaurant','manager','retail','buyer',
                   'retail','jeweller','retail','manager','retail','merchandiser','riding','holiday','centre','manager','riding',
                   'holiday','leader','road','haulage','load','planner','road','traffic','accident','investigator','road','transport',
                   'manager','road','worker','roadie','roofer','roundsperson','royal','marines','commando','royal','marines','officer',
                   'royal','navy','officer','royal','navy','rating','rspca','inspector','rural','surveyor','sailing','instructor','sales',
                   'assistant','sales','manager','sales','promotion','executive','sales','representative','sample','machinist','satellite',
                   'systems','technician','scaffolder','scenes','of','crime','officer','school','business','manager','school','crossing',
                   'patrol','school','lunchtime','supervisor','school','matron','school','nurse','school','secretary','screenwriter',
                   'secondary','school','teacher','secretary','security','officer','security','service','personnel','security',
                   'systems','installer','set','designer','sewing','machinist','sexual','health','adviser','sheet','metal','worker',
                   'shoe','repairer','shopfitter','shopkeeper','signalling','technician','signmaker','signwriter','singing','teacher',
                   'skills','for','life','teacher','smart','meter','installer','social','media','manager','social','work','assistant',
                   'social','worker','software','developer','solicitor','special','educational','needs','teacher','special','needs',
                   'teaching','assistant','speech','and','language','therapist','speech','and','language','therapy','assistant','sport',
                   'and','exercise','psychologist','sport','and','exercise','scientist','sports','coach','sports','commentator','sports',
                   'development','officer','sports','physiotherapist','sports','professional','stage','manager','stagehand','steel','erector',
                   'steel','fixer','steeplejack','or','lightning','conductor','engineer','sterile','services','technician','stockbroker',
                   'stonemason','store','demonstrator','store','detective','structural','engineer','studio','sound','engineer','stunt',
                   'performer','sub-editor','substance','misuse','outreach','worker','supervisor','supply','chain','manager','surgeon',
                   'swimming','pool','technician','swimming','teacher','or','coach','systems','analyst','tailor','tanker','driver','tattooist',
                   'tax','adviser','tax','inspector','taxi','driver','teacher','teacher','teacher','teacher','teacher','teaching','assistant',
                   'technical','architect','or','it','systems','architect','technical','author','technical','brewer','technical','surveyor',
                   'technical','textiles','designer','telecoms','technician','telephonist-switchboard','operator','textile','designer',
                   'textile','dyeing','technician','textile','machinery','technician','textile','operative','textile','technologist',
                   'textiles','production','manager','thatcher','thermal','insulation','engineer','tiler','timber','yard','worker',
                   'toolmaker','tour','manager','tourist','guide','tourist','information','centre','assistant','town','planner','town',
                   'planning','support','staff','trade','mark','attorney','trade','union','official','trading','standards','officer',
                   'train','conductor','train','driver','train','station','staff','training','manager','training','officer','tram',
                   'driver','translator','transport','planner','travel','agent','tree','surgeon','tv','or','film','assistant',
                   'director','tv','or','film','camera','operator','tv','or','film','director','tv','or','film','producer','tv',
                   'or','film','production','assistant','tv','or','film','production','manager','tv','or','film','sound','technician',
                   'tv','presenter','tv','production','runner','typist','upholsterer','vending','machine','operative','veterinary',
                   'nurse','veterinary','physiotherapist','veterinary','surgeon','victim','care','officer','video','editor','visitor',
                   'attraction','general','manager','visual','merchandiser','volunteer','organiser','waiting','staff','wardrobe',
                   'assistant','warehouse','manager','warehouse','operative','waste','management','officer','watch','or','clock',
                   'repairer','water','network','operative','water','treatment','worker','web','content','manager','web','designer',
                   'web','developer','web','editor','wedding','planner','welder','welfare','rights','officer','window','cleaner',
                   'window','fitter','wine','merchant','wood','machinist','writer','yard','person','yoga','teacher','yoga','therapist',
                   'youth','and','community','worker','youth','offending','team','officer','zookeeper','zoologist','ceo','cto', 'cfo']

occupation_list = set(occupation_list)

def reading_data(file_name):
    global training_title
    with open(file_name, 'rb') as f:
        data_set = pickle.load(f)
    return data_set

def extract_classify_result(data_set):
    classify_result=[]
    for sentence in data_set:
        for word in sentence:
            classify_result.append(word[2])
    return classify_result

def convert_word_to_feature_dict(set):
    feature_vector_dict = []
    for sen in set:
        for index in range(len(sen)):
            feature_dict = extract_feature(sen[index], sen, index)
            feature_vector_dict.append(feature_dict)
    return feature_vector_dict

def extract_feature(word, sentence, index):
    feature_vector_dict = {}
    if index > 0:
        feature_vector_dict['type_' + sentence[index - 1][0]] = 2
    if index < len(sentence)-1:
        feature_vector_dict['type_' + sentence[index + 1][0]] = 2
    if word[0].lower() in abravation:
        feature_vector_dict['type_abravation'] = 1
    feature_vector_dict['type'+word[1]] = 1
    if word[0].lower() in head_list and (index<len(sentence)-1) and (sentence[index+1][0].lower() in occupation_list):
        feature_vector_dict['type_next_title'+sentence[index+1][0].lower()] = 1
    weight = 0
    if word[0].lower() in occupation_list:
        feature_vector_dict[word[0].lower()]=1
        weight +=1
        itr = index
        itr = index
        while itr > 0 and (sentence[itr - 1][0].lower() in occupation_list):
            weight += 1
            itr -= 1
            feature_vector_dict['type_previous'+str(weight)+sentence[itr][0].lower()] = 1
        itr = index
        while itr < len(sentence) - 1 and (sentence[itr + 1][0].lower() in occupation_list):
            weight += 1
            itr+=1
            feature_vector_dict['type_next'+str(weight)+sentence[itr][0].lower()] = 1
    feature_vector_dict['type_phrase'] = weight
    feature_vector_dict[word[0].lower()] =  1
    return feature_vector_dict

if __name__ == '__main__':
    path_to_training_data = sys.argv[1]
    path_to_classifier = sys.argv[2]
    data_set = reading_data(path_to_training_data)
    classify_result = np.array(extract_classify_result(data_set))
    temp_feature_vector_dict = convert_word_to_feature_dict(data_set)
    vec = DictVectorizer()
    feature_vector_dict = vec.fit_transform(temp_feature_vector_dict).toarray()
    clf = LogisticRegression(penalty='l1')
    clf.fit(feature_vector_dict,classify_result)
    with open(path_to_classifier, 'wb') as f:
        pickle.dump((clf, vec), f)
    
