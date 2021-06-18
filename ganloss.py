
import torch
import torch.nn as nn
import torch.nn.functional as F

class _GANLoss() :
	def __init__( self, device ) :
		self.device = device
	
	def __call__( self, logits, labels, mask = None, size_average = True ) :
		self.size_average = size_average
		if isinstance( logits, list ) :
			if size_average :
				loss = sum( [ self.run( logit, labels, mask, dom = len( logits ) ) for logit in logits ] )
			else :
				loss = sum( [ self.run( logit, labels, mask ) for logit in logits ] )
		else :
			loss = self.run( logits, labels, mask )
		return loss

class GANLossSCE( _GANLoss ) :
	def __init__( self, device ) :
		super( GANLossSCE, self ).__init__( device )
		self.criterion = nn.BCEWithLogitsLoss()

	def run( self, logits, labels, mask, dom = 1 ) :
		bs = logits.size( 0 )
		if labels == 'real' :
			lbl = torch.ones( logits.size(), device = self.device )
		if labels == 'fake' :
			lbl = torch.zeros( logits.size(), device = self.device )
		if labels == 'generator' :
			lbl = torch.ones( logits.size(), device = self.device )
		loss = self.criterion( logits, lbl ) / dom
		return loss

class GANLossLS( _GANLoss ) :
	def __init__( self, device ) :
		super( GANLossLS, self ).__init__( device )
		self.criterion = nn.MSELoss()

	def run( self, logits, labels, mask, dom = 1 ) :
		bs = logits.size( 0 )
		if labels == 'real' :
			lbl = torch.ones( logits.size(), device = self.device )
		if labels == 'fake' :
			lbl = torch.zeros( logits.size(), device = self.device )
		if labels == 'generator' :
			lbl = torch.ones( logits.size(), device = self.device )
		loss = self.criterion( logits, lbl ) / dom
		return loss

class GANLossSoftLS( _GANLoss ) :
	def __init__( self, device ) :
		super( GANLossSoftLS, self ).__init__( device )
		self.criterion = nn.MSELoss()

	def run( self, logits, labels, mask, dom = 1 ) :
		bs = logits.size( 0 )
		if labels == 'real' :
			lbl = torch.ones( logits.size(), device = self.device )
		if labels == 'fake' :
			lbl = mask
		if labels == 'generator' :
			lbl = torch.ones( logits.size(), device = self.device )
		loss = self.criterion( logits, lbl ) / dom
		if labels == 'fake' :
			loss = loss / mask.mean()
		return loss

class GANLossHinge( _GANLoss ) :
	def __init__( self, device ) :
		super( GANLossHinge, self ).__init__( device )

	def run( self, logits, labels, mask, dom = 1 ) :      
		if labels == 'real' :
			loss = nn.ReLU()( torch.ones( logits.size(), device = self.device ) - logits )
		if labels == 'fake' :
			loss = nn.ReLU()( torch.ones( logits.size(), device = self.device ) + logits )
		if labels == 'generator' :
			loss = -logits
		loss = loss.mean() / dom
		return loss

class GANLossRaSCE( _GANLoss ) :
	def __init__( self, device ) :
		super( GANLossRaSCE, self ).__init__( device )
		self.criterion = nn.BCEWithLogitsLoss()

	def dloss( self, reals, fakes ) :
		lbl_real = torch.ones( reals.size(), device = self.device )
		lbl_fake = torch.zeros( reals.size(), device = self.device )
		avg_reals = reals.mean( dim = 0 )
		avg_fakes = fakes.mean( dim = 0 )
		return 0.5 * ( self.criterion( reals - avg_fakes, lbl_real ) + self.criterion( fakes - avg_reals, lbl_fake ) )
	
	def gloss( self, reals, fakes ) :
		lbl_real = torch.ones( reals.size(), device = self.device )
		lbl_fake = torch.zeros( reals.size(), device = self.device )
		avg_reals = reals.mean( dim = 0 )
		avg_fakes = fakes.mean( dim = 0 )
		return 0.5 * ( self.criterion( fakes - avg_reals, lbl_real ) + self.criterion( reals - avg_fakes, lbl_fake ) )

class GANLossRaHinge( _GANLoss ) :
	def __init__( self, device ) :
		super( GANLossRaHinge, self ).__init__( device )
		self.criterion = nn.BCEWithLogitsLoss()

	def dloss( self, reals, fakes ) :
		avg_reals = reals.mean( dim = 0 )
		avg_fakes = fakes.mean( dim = 0 )
		Dreal = reals - avg_fakes
		Dfake = fakes - avg_reals
		return 0.5 * ( F.relu( 1 - Dreal ) + F.relu( 1 + Dfake ) ).mean()
	
	def gloss( self, reals, fakes ) :
		avg_reals = reals.mean( dim = 0 )
		avg_fakes = fakes.mean( dim = 0 )
		Dreal = reals - avg_fakes
		Dfake = fakes - avg_reals
		return 0.5 * ( F.relu( 1 - Dfake ) + F.relu( 1 + Dreal ) ).mean()
	
class GANLossQP( _GANLoss ) :
	def __init__( self, device ) :
		super( GANLossQP, self ).__init__( device )

	def run( self, logits, labels, mask, dom = 1 ) :
		if labels == 'real' :
			loss = -logits
		if labels == 'fake' :
			loss = logits
		if labels == 'generator' :
			loss = -logits
		loss = loss.mean() / dom
		return loss

	def qp( self, real, fake, real_logits, fake_logits, lambda_ = 10. ) :
		dim = real.shape[1] * real.shape[2] * real.shape[3]
		dis = torch.mean( torch.abs( real - fake ).view( -1, dim ), dim = 1 )
		d_loss = real_logits - fake_logits
		d_loss = d_loss - .5 * ( d_loss ** 2 ) / ( lambda_ * dis )
		return -d_loss.mean()


