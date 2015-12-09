/***BEFORE***/

for(map<string, pair<vector<Emotion>, Classifier*> >::iterator ii =
     this->detectors_ext.begin(); ii != this->detectors_ext.end(); ++ii) {

   vector<Emotion> emo = ii->second.first; // detected emotions
   Classifier* cl = ii->second.second;

   float prediction = cl->predict(frame);

   for(vector<Emotion>::iterator emo_it = emo.begin(); emo_it != emo.end(); ++emo_it) {
     map<Emotion, float>::iterator it = votes.find(*emo_it);
     if (it == votes.end()) {
       votes.insert(make_pair(*emo_it, prediction));
     } else{
       if (prediction > 0.5) {
         it->second += 0.5; //1.0;
       } else {
         it->second -= 0.0;
       }
     }
   }
 }

/***AFTER***/

vector<pair<string, pair<vector<Emotion>, Classifier*> > > v(detectors_ext.begin(), detectors_ext.end());
#pragma omp parallel for
for(int y=0; y<v.size(); y++){
  pair<string, pair<vector<Emotion>, Classifier*> > p = v[y];
  vector<Emotion> emo = p.second.first; // detected emotions
  Classifier* cl = p.second.second;

  float prediction = cl->predict(frame);

  for(vector<Emotion>::iterator emo_it = emo.begin(); emo_it != emo.end(); ++emo_it) {
    map<Emotion, float>::iterator it = votes.find(*emo_it);
    if (it == votes.end()) {
      votes.insert(make_pair(*emo_it, prediction));
    } else{
      if (prediction > 0.5) {
        it->second += 0.5; //1.0;
      } else {
        it->second -= 0.0;
      }
    }
  }
