"""Synthetic visual evidence and API pagination regressions; no models/images."""
import contextlib
import sqlite3

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from api.routers import gallery


@pytest.fixture
def db():
 c=sqlite3.connect(':memory:',check_same_thread=False);c.row_factory=sqlite3.Row
 c.execute('CREATE TABLE photos(path TEXT PRIMARY KEY,filename TEXT,phash TEXT,clip_embedding BLOB,date_taken TEXT,aggregate REAL,aesthetic REAL,is_visible INTEGER DEFAULT 1,histogram_data BLOB,mean_saturation REAL,mean_luminance REAL,is_monochrome INTEGER)')
 yield c;c.close()
def add(db,path,phash=None,embedding=None,visible=1):
 if embedding is not None and not isinstance(embedding,bytes):embedding=np.array(embedding,dtype=np.float32).tobytes()
 db.execute('INSERT INTO photos(path,filename,phash,clip_embedding,is_visible) VALUES(?,?,?,?,?)',(path,path,phash,embedding,visible));db.commit()
def search(db,source,threshold=0):return gallery._find_similar_visual(db,source,'source',threshold,'is_visible=1',[])

def test_phash_only_full_similarity(db):
 add(db,'same','0');r,_=search(db,{'phash':'0'},.95);assert r[0]['similarity']==1

def test_clip_only_source_and_candidates_not_hidden_by_other_phash(db):
 add(db,'phash-only','f');add(db,'embedding-only',None,[1,0]);r,_=search(db,{'clip_embedding':np.array([1,0],dtype=np.float32).tobytes()});assert [x['path'] for x in r]==['embedding-only']

def test_pairwise_available_weights(db):
 add(db,'hash','0');add(db,'clip',None,[1,0]);add(db,'both','0',[0,1]);r,_=search(db,{'phash':'0','clip_embedding':np.array([1,0],dtype=np.float32).tobytes()});assert {v['path']:v['similarity'] for v in r}=={'hash':1.,'clip':1.,'both':.7}

@pytest.mark.parametrize('broken',[b'x',np.array([float('nan'),0],dtype=np.float32).tobytes(),np.array([1,0,0],dtype=np.float32).tobytes(),bytes(8)])
def test_bad_embedding_never_aborts_or_fake_matches_but_preserves_hash(db,broken):
 add(db,'hash-fallback','0',broken);add(db,'invalid-only',None,broken);r,_=search(db,{'phash':'0','clip_embedding':np.array([1,0],dtype=np.float32).tobytes()});assert [v['path'] for v in r]==['hash-fallback']

def test_invalid_hash_unknown_excluded_at_zero(db):
 add(db,'bad','z');add(db,'negative','-1');add(db,'too-wide','1'*17);add(db,'unknown');assert search(db,{'phash':'0'})[0]==[]

def test_missing_source_is_explicit_unavailable(db):
 add(db,'indexed','0');r,message=search(db,{});assert r==[] and 'unavailable' in message

def test_identical_float32_embeddings_at_100_percent_boundary(db):
 rng=np.random.default_rng(42)
 for i in range(100):
  vector=rng.normal(size=16).astype(np.float32).tobytes();db.execute('DELETE FROM photos');add(db,'identical',None,vector)
  r,_=search(db,{'clip_embedding':vector},1.0);assert r[0]['similarity']==1.0
 db.execute('DELETE FROM photos');add(db,'near-but-not-identical',None,[1,.001])
 assert search(db,{'clip_embedding':np.array([1,0],dtype=np.float32).tobytes()},1.0)[0]==[]

def test_full_set_not_silent_500_and_stable_visibility(db):
 for i in range(603):add(db,f'candidate-{i:04}','0')
 add(db,'hidden','0',visible=0);add(db,'unindexed')
 r,_=search(db,{'phash':'0'});assert len(r)==603 and [v['path'] for v in r]==sorted(v['path'] for v in r)

@pytest.mark.parametrize('threshold',[-.1,1.1,float('nan'),float('inf')])
def test_invalid_threshold_rejected(db,threshold):
 with pytest.raises(ValueError):search(db,{'phash':'0'},threshold)

class Cursor:
 def __init__(self,c):self.c=c
 async def fetchone(self):return self.c.fetchone()
 async def close(self):self.c.close()
class AsyncDB:
 def __init__(self,c):self.c=c
 async def execute(self,*a):return Cursor(self.c.execute(*a))
def client(db,monkeypatch):
 @contextlib.asynccontextmanager
 async def async_db():yield AsyncDB(db)
 @contextlib.contextmanager
 def sync_db():yield db
 app=FastAPI()
 app.include_router(gallery.router)
 app.dependency_overrides[gallery.get_optional_user]=lambda:None
 monkeypatch.setattr(gallery,'get_async_db',async_db)
 monkeypatch.setattr(gallery,'get_db',sync_db)
 monkeypatch.setattr(gallery,'get_visibility_clause',lambda user_id:('is_visible=1',[]))
 monkeypatch.setitem(gallery.VIEWER_CONFIG,'features',{'show_similar_button':True})
 return TestClient(app)

def test_actual_api_zero_total_pagination_over_500(db,monkeypatch):
 add(db,'source','0')
 for i in range(603):add(db,f'candidate-{i:04}','f'*16)
 with client(db,monkeypatch) as c:
  a=c.get('/api/similar_photos/source',params={'min_similarity':0,'limit':500}).json();b=c.get('/api/similar_photos/source',params={'min_similarity':0,'limit':500,'offset':500}).json();positive=c.get('/api/similar_photos/source',params={'min_similarity':.01}).json()
 assert positive['total']==0 and all(v['similarity']==0 for v in a['similar'])
 assert a['total']==b['total']==603 and a['has_more'] and not b['has_more'] and len(a['similar'])==500 and len(b['similar'])==103
 assert not {v['path'] for v in a['similar']} & {v['path'] for v in b['similar']}

@pytest.mark.parametrize('params',[{'offset':-1},{'limit':0},{'limit':501},{'min_similarity':-.1},{'min_similarity':1.1},{'min_similarity':'nan'},{'min_similarity':'inf'}])
def test_actual_api_query_validation(db,params,monkeypatch):
 add(db,'source','0')
 with client(db,monkeypatch) as c:assert c.get('/api/similar_photos/source',params=params).status_code==422
