
use numrs::autograd::train::Trainer;
use numrs::autograd::optim::{
    SGD, Adam, AdamW, NAdam, RAdam, LAMB, AdaBound, 
    RMSprop, AdaGrad, AdaDelta,
    LBFGS, Rprop
};
use numrs::autograd::Sequential;

pub mod sgd;
pub mod adam;
pub mod rms;
pub mod others;

pub enum TrainerWrapper {
    Sgd(Trainer<Sequential, SGD>),
    Adam(Trainer<Sequential, Adam>),
    AdamW(Trainer<Sequential, AdamW>),
    NAdam(Trainer<Sequential, NAdam>),
    RAdam(Trainer<Sequential, RAdam>),
    Lamb(Trainer<Sequential, LAMB>),
    AdaBound(Trainer<Sequential, AdaBound>),
    RmsProp(Trainer<Sequential, RMSprop>),
    AdaGrad(Trainer<Sequential, AdaGrad>),
    AdaDelta(Trainer<Sequential, AdaDelta>),
    Lbfgs(Trainer<Sequential, LBFGS>),
    Rprop(Trainer<Sequential, Rprop>),
}

use crate::train::NumRsDataset;

impl TrainerWrapper {
    pub fn fit(&mut self, dataset: &NumRsDataset, epochs: usize) {
         let ds = &dataset.inner;
         match self {
             Self::Sgd(t) => { let _ = t.fit(ds, None, epochs, true); },
             Self::Adam(t) => { let _ = t.fit(ds, None, epochs, true); },
             Self::AdamW(t) => { let _ = t.fit(ds, None, epochs, true); },
             Self::NAdam(t) => { let _ = t.fit(ds, None, epochs, true); },
             Self::RAdam(t) => { let _ = t.fit(ds, None, epochs, true); },
             Self::Lamb(t) => { let _ = t.fit(ds, None, epochs, true); },
             Self::AdaBound(t) => { let _ = t.fit(ds, None, epochs, true); },
             Self::RmsProp(t) => { let _ = t.fit(ds, None, epochs, true); },
             Self::AdaGrad(t) => { let _ = t.fit(ds, None, epochs, true); },
             Self::AdaDelta(t) => { let _ = t.fit(ds, None, epochs, true); },
             Self::Lbfgs(t) => { let _ = t.fit(ds, None, epochs, true); },
             Self::Rprop(t) => { let _ = t.fit(ds, None, epochs, true); },
         }
    }
}
