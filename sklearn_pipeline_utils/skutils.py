import pandas as pd
import numpy as np

from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn_pandas import DataFrameMapper

from sklearn.preprocessing import LabelBinarizer

# Column Etracter


def ColumnExtracter(columns):
	'''
	params :

	columns		list of strings, column names to extract
	'''
	def extractColumns(X, columns):
		return X[columns]
	return FunctionTransformer(
		extractColumns,
		validate=False,
		kw_args={'columns': columns}
	)


# Imputer

class CustomImputer(BaseEstimator, TransformerMixin):
	'''
	A wrapper over the sklearn.preprocessing.Imputer to handle all kinds of data
	'''
	def __init__(self, strategy='mean', filler='NA'):
		'''
		params :

		strategy 	mean - imputes with mean of the column values, numerical values required
					median - imputes with median of the column values, numerical values required
					mode - imputes with mode of the column values
					strategy - imputed with the provided or default fill value

		filler		can be a sigle value, string or number
					list of values whose length is equal to the number of columns can be provided
					which are used to impute each column
		'''
		self.strategy = strategy
		self.fill = filler

	def fit(self, X, y=None):
		if self.strategy in ['mean', 'median']:
			if not all([dtype in [np.number, np.int] for dtype in X.dtypes]):
				raise ValueError('dtypes mismatch np.number dtype is required for ' + self.strategy)
		if self.strategy == 'mean':
			self.fill = X.mean()
		elif self.strategy == 'median':
			self.fill = X.median()
		elif self.strategy == 'mode':
			self.fill = X.mode().iloc[0]
		elif self.strategy == 'fill':
			if type(self.fill) is list and type(X) is pd.DataFrame:
				self.fill = dict([(cname, v) for cname, v in zip(X.columns, self.fill)])
		return self

	def transform(self, X, y=None):
		if self.fill is None:
			self.fill = 'NA'
		return X.fillna(self.fill)


# ColumnsEqualityChecker

def ColumnsEqualityChecker(result_column='equality_col', inverse=False):
	'''
	checks if values in each row across columns are the same

	params:

	result_column	column name of the result column
	inverse			apply a not function to the result, checks inequality
	'''
	def equalityChecker(X, result_column, inverse=False):
		def roweq(row):
			eq = all(row.values == row.values[0])
			return eq
		eq = X.apply(roweq, axis=1)
		if inverse:
			eq = eq.apply(np.invert)
		return pd.DataFrame(eq.values.astype(int), columns=[result_column])
	return FunctionTransformer(
		equalityChecker,
		validate=False,
		kw_args={'result_column': result_column, 'inverse': inverse}
	)


# CustomMapper

def CustomMapper(result_column='mapped_col', value_map={}, default=np.nan):
	'''
	maps the values of the columns accroding to given mapping

	params:

	result_column	name of the resulting column, suffixed _n if more than a
					column is returned

	value_map		dict with the mapping of col_value: to_be_mapped_value

	default			default mapped_value for column values not in value_map
	'''
	def mapper(X, result_column, value_map, default):
		def colmapper(col):
			return col.apply(lambda x: value_map.get(x, default))
		mapped_col = X.apply(colmapper).values
		mapped_col_names = [result_column + '_' + str(i) for i in range(mapped_col.shape[1])]
		return pd.DataFrame(mapped_col, columns=[mapped_col_names])
	return FunctionTransformer(
		mapper,
		validate=False,
		kw_args={'result_column': result_column, 'value_map': value_map, 'default': default}
	)


# Column Sum

def ColumnsSum(result_column='sum_col'):
	'''
	returns sum of dataframe row by row

	params:

	result_column	name of the result column
	'''
	def colSum(X, result_column):
		return pd.DataFrame(X.sum(axis=1), columns=[result_column])
	return FunctionTransformer(
		colSum,
		validate=False,
		kw_args={'result_column': result_column}
	)

# Categorical Encoder


def ColumnsLabelBinarizer(columns):
	'''
	wrapper over the sklearn.preprocessing.LabelBinarizer to handle categorical data

	params:

	columns 	names of columns to LabelBinarize
	'''
	return DataFrameMapper(
		[(col, LabelBinarizer()) for col in columns],
		df_out=True,
		default=None
	)


# ColumnValuetoBoolean


def ColumnValueToBoolean(positive_string, inverse=False):
	'''
	To check if the value of the column equals a specific value, returns a binarized column with the truth value

	params:

	positive_string		value to look for in the columns

	inverse				apply a not operation to the result, can be used to extract all the values not equal to a
						specific value in the column
	'''
	def valueToBoolean(X, positive_string, inverse=False):
		result = (X == positive_string)
		if inverse:
			result = np.invert(result)
		return result.astype(int)
	if not positive_string:
		raise ValueError('positive_string is not specified')
	return FunctionTransformer(
		valueToBoolean,
		validate=False,
		kw_args={'positive_string': positive_string, 'inverse': inverse}
	)


# DataFramePrinter


def DataFramePrinter(columns_only=False, shape_only=False):
	'''
	A util to print dataframe details in the middle of a pipeline

	params:

	columns_only	print only the column names

	shape_only		print only the shape of the dataframe
	'''
	def dfprinter(X):
		if columns_only:
			print X.columns
		elif shape_only:
			print X.shape
		else:
			print X.head()
		return X
	return FunctionTransformer(dfprinter, validate=False)
