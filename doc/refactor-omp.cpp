/***BEFORE***/

for (map<string, pair<vector<Emotion>, Classifier*> >::iterator ii =
    this->detectors_ext.begin(); ii != this->detectors_ext.end(); ++ii) {
  if (ii->second.first.size() != 1) {
    continue;
  }
  Emotion emo = ii->second.first[0];
  Classifier* cl = ii->second.second;
  float prediction = cl->predict(frame);
  map<Emotion, float>::iterator it = votes.find(emo);
  if (it == votes.end()) {
    votes.insert(make_pair(emo, prediction));
  } else {
    it->second+=prediction;
  }
}

/***AFTER***/

vector<pair<string, pair<vector<Emotion>, Classifier*> > > v(detectors_ext.begin(), detectors_ext.end());
#pragma omp parallel for
for(int y=0; y<v.size(); y++){
  pair<string, pair<vector<Emotion>, Classifier*> > p = v[y]
  if (p.second.first.size() != 1) {
    continue;
  }
  Emotion emo = p.second.first[0];
  Classifier* cl = p.second.second;
   float prediction = cl->predict(frame);
   map<Emotion, float>::iterator it = votes.find(emo);
  if (it == votes.end()) {
    votes.insert(make_pair(emo, prediction));
  } else {
    it->second+=prediction;
  }
}

