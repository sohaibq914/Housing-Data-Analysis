import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
import pandas as pd
import random

"""
Explanation:
This project is used to analyze housing data and from that
we found what factors influence the house prices the most
"""

# This dataset shows different housing incomes, prices,
# and attributes that could be taken into account when using this data for specific tasks.
# This is only for California-based housing,
# where each house has its proximity to the ocean and Bay Area location.
def housing_one(show=False):
    '''
    Show the scatter plots for all columns on the same plot + do pivot tables with housing prices

    :param show: show plot or not
    :return: the final dataframe
    '''
    # read csv
    housing_one = pd.read_csv('Housing.csv')

    # sort values by the median house value in ascending order to make graphs make more sense
    housing_one.sort_values(by='median_house_value', inplace=True)

    # since there are two many of the same housing median age values, we remove duplicates to maek graphs more succinct
    housing_one = housing_one[housing_one.duplicated('housing_median_age') == False]

    # make 8 subplots for each factor
    rows = 2
    cols = 3
    fig, ax = plt.subplots(nrows=rows, ncols=cols, sharex=True)

    columns = {'housing_median_age': ['red', 'o'],
               'total_rooms': ['orange', 'x'],
               'total_bedrooms': ['yellow', '^'],
               'population': ['green', '*'],
               'households': ['blue', 'P'],
               'median_income': ['magenta', '+']}

    count = 0

    # loop through subplots
    for row in range(rows):
        for col in range(cols):
            keys_as_list = list(columns.keys())

            # plot as long as color is not white
            ax[row][col].scatter(list(range(len(housing_one))),
                                 housing_one[keys_as_list[count]],
                                 color=columns[keys_as_list[count]][0],
                                 marker=columns[keys_as_list[count]][1])
            # set title
            ax[row][col].set_title(keys_as_list[count])

            # increment to move on to next column
            count += 1

    # tight layout just to make sure everything is spaced fairly
    plt.tight_layout()

    # show (not for jupyter notebook)
    if show:
        plt.show()

    # can't run aggfunc on a string, so we might drop ocean_proximity if it's not an index (ex. NEAR BAY, OCEAN, etc.)
    # take mean of median house values
    new_df = housing_one.set_index(['ocean_proximity'])
    new_df.sort_index(inplace=True)

    # filter longitude and latitude
    new_df = new_df[(new_df['longitude'] > -122) & (new_df['longitude'] < -120) &
                    (new_df['latitude'] > 35) & (new_df['latitude'] < 40)]

    # for only inland proximity for specific latitude/longitude ranges, view housing median ages
    new_df = new_df.pivot_table(index=['ocean_proximity'], columns=['median_house_value'])

    # print it with the house price being the column
    print(new_df)

    # stack it so that the proximity and house price line up
    new_df = new_df.stack()

    print(new_df)

    return housing_one


# Urban housing listed on Craigslist for the entire United States population’s housing.
# It has its own column for location, as well as whether the house allows cats and its corresponding price.
def housing_two(show=False):
    # read in csv
    housing_two = pd.read_csv('Housing2.csv')

    def kde(df):
        '''
        Density plot denoting housing prices

        :param df: the dataframe to display
        :return: the dataframe in question
        '''
        # remove all the duplicate locations to have one of each state (and, by extension, region) for diversity
        df = df[df.duplicated(subset=['state'], keep='first') == False]

        # remove columns with url, it is not necessary for our purposes
        df.drop(labels=['url', 'region_url'], axis=1, inplace=True)

        # subplots
        fig, ax = plt.subplots(1, 2)

        # density plot
        df['price'].plot.kde(ax=ax[0], title='Density plot')
        ax[0].set_xlabel('Prices')

        # violin plot
        ax[1].violinplot(df['price'], showmedians=True)
        ax[1].set_xlabel('Density')
        ax[1].set_ylabel('Prices')
        ax[1].set_title('Violin plot')

        # tight layout for multiple plots
        fig.tight_layout()
        if show:
            plt.show()

        return df

    def scatter_plot(df, legend=False):
        """
        Plots prices across how large each house is (square feet) across unique regions with random colors and markers.

        :return: n/a
        """

        # regions with color and markers
        # regions are already unique as duplicates were previously removed
        regions = dict()

        # gather index + state pair
        for i, state in enumerate(df['region']):
            regions[state] = []
            # append random color to first element of value
            regions[state].append(random.choice(list(mcolors.CSS4_COLORS)))
            # choose random marker
            marker = random.choice(list(Line2D.markers))
            # append marker to second element
            regions[state].append(marker)

        for r in regions:
            # plot regions and price based on dictionary list values
            plt.scatter(
                df[df['region'] == r]['sqfeet'],
                df[df['region'] == r]['price'],
                color=regions[r][0],
                marker=regions[r][1])

        # no legend as there are too many keys
        if legend:
            plt.legend(regions.keys())
        plt.xlabel('House Dimensions (Square Feet)')
        plt.ylabel('House Price ($)')
        plt.title('House Price ($) vs. House Dimensions (Square feet)')
        if show:
            plt.show()

    # call kde plot
    df = kde(housing_two)

    # call scatter plot with whole dataset
    scatter_plot(df)

    # call scatterplot with only a couple
    filtered_regions = housing_two.loc[0:100, :]
    scatter_plot(filtered_regions, legend=True)

    return housing_two



# This dataset contains 13 features that can be used to predict the price of housing.
# It states whether each house is on the main road, guestroom, basement, hot water heating, or air conditioning.
def housing_three(show=True):
    '''
    Display housing three data
    :param show: show the plots or not
    :return: the final dataframe
    '''
    housing_three = pd.read_csv("Housing3.csv")
    housing_three.describe()  # convert to percentiles, mean, std, and more for boxplots
    housing_three.isnull().sum()  # 0 across all categories indicating no null values

    unique_statuses = housing_three['furnishingstatus'].unique()  # gets the different types statuses
    data_by_status = [] # store the data for each status
    for status in unique_statuses:  # iterating over every status
        price_data_for_status = housing_three[housing_three['furnishingstatus'] == status][
            'price']  # getting the all the prices for each df
        data_by_status.append(price_data_for_status)  # store df for each status into the array

    plt.figure(figsize=(8, 6))
    plt.xticks(rotation=45)
    plt.boxplot(data_by_status, labels=unique_statuses)

    # setting titles and labels
    plt.title('Influence of Furnishing Status on House Prices')
    plt.xlabel('Furnishing Status')
    plt.ylabel('House Price')
    if show:
        plt.show()

    # these the bins ranges
    sq_feet_bins = [0, 2500, 5000, 7500, 10000]

    housing_three['sq_feet_range'] = pd.cut(housing_three['area'], bins=sq_feet_bins)

    plt.figure(figsize=(8, 6))
    plt.xticks(rotation=45)
    data_by_sq_feet_range = []

    for range_val in housing_three['sq_feet_range'].unique():  # iterate over unique categories in col
        # getting all the prices for each range
        price_data_for_range = housing_three[housing_three['sq_feet_range'] == range_val]['price']

        # store each df for each range into the array
        data_by_sq_feet_range.append(price_data_for_range)

    plt.boxplot(data_by_sq_feet_range, labels=housing_three['sq_feet_range'].cat.categories)

    plt.title('Relationship between House Prices and Square Feet')
    plt.xlabel('Square Feet Range')
    plt.ylabel('House Price')
    if show:
        plt.show()

    plt.figure(figsize=(10, 6))
    plt.xticks(rotation=0)

    data_by_bedrooms = []
    housing_three.sort_values(by='bedrooms', inplace=True)  # sort so bedroom # in boxplots are in order

    for num_bedrooms in housing_three['bedrooms'].unique():  # iterating over num of bedrooms
        # getting all prices for that num of bedrooms
        price_data_for_bedrooms = housing_three[housing_three['bedrooms'] == num_bedrooms]['price']
        # store each df for each bedroom num into the array
        data_by_bedrooms.append(price_data_for_bedrooms)

    plt.boxplot(data_by_bedrooms, labels=housing_three['bedrooms'].unique())
    plt.ylabel('House Price (in millions)')
    plt.title('Relationship between House Prices and Number of Bedrooms')
    if show:
        plt.show()

    # do same thing with bathroom
    data_by_bathrooms = []
    housing_three.sort_values(by='bathrooms', inplace=True)  # sort so bedroom # in boxplots are in order

    for num_bathrooms in housing_three['bathrooms'].unique():
        # retrieved price after filtering by bathroom
        price_data_for_bathrooms = housing_three[housing_three['bathrooms'] == num_bathrooms]['price']

        data_by_bathrooms.append(price_data_for_bathrooms)

    plt.boxplot(data_by_bathrooms, labels=housing_three['bathrooms'].unique())

    plt.ylabel('House Price (in millions)')
    plt.title('Relationship between House Prices and Number of Bathrooms')
    if show:
        plt.show()

    plt.figure(figsize=(10, 6))

    # calculate mean price for each furnishing status
    furnishing_status_prices = housing_three.groupby('furnishingstatus')['price'].mean()

    furnishing_status_prices.plot(kind='bar', color="blue")

    plt.xlabel('Furnishing Status')
    plt.ylabel('Average Housing Price (in hundred thousands)')
    plt.title('Average Housing Prices by Furnishing Status')
    plt.xticks(rotation=0)  # Keep x-axis labels horizontal for better readability
    if show:
        plt.show()

    # calculate mean price for each num of bedrooms
    bedroom_prices = housing_three.groupby('bedrooms')['price'].mean()
    plt.figure(figsize=(10, 6))
    bedroom_prices.plot(kind='bar', color="green")
    plt.xlabel('Bedrooms #')
    plt.ylabel('Average Housing Price (in hundred thousands)')
    plt.title('Average Housing Prices By Bedrooms')
    plt.xticks(rotation=45)
    if show:
        plt.show()

    # calculate mean price for each num of bedrooms
    bathroom_prices = housing_three.groupby('bathrooms')['price'].mean()
    plt.figure(figsize=(10, 6))
    bathroom_prices.plot(kind='bar', color="green")
    plt.xlabel('# of Bathrooms')
    plt.ylabel('Average Housing Price (in hundred thousands)')
    plt.title('Average Housing Prices By Bathrooms')
    plt.xticks(rotation=45)
    if show:
        plt.show()

    # same functionality
    average_price_by_furnishing = housing_three.groupby('furnishingstatus')['price'].mean()
    plt.figure(figsize=(10, 6))

    # plot line chart
    plt.plot(average_price_by_furnishing.index, average_price_by_furnishing.values, marker='o', linestyle='-', color="blue")

    plt.xlabel('Furnishing Status')
    plt.ylabel('Average Housing Price (in hundred thousands)')
    plt.title('Average Housing Prices by Furnishing Status')

    plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility if needed

    if show:
        plt.show()

    # same functionality
    average_price_by_bedrooms = housing_three.groupby('bedrooms')['price'].mean()
    plt.figure(figsize=(10, 6))

    # plot line chart
    plt.plot(average_price_by_bedrooms.index, average_price_by_bedrooms.values, marker='o', linestyle='-', color="green")

    plt.xlabel('# of Bedrooms')
    plt.ylabel('Average Housing Price (in hundred thousands)')
    plt.title('Average Housing Prices By Bedrooms')

    if show:
        plt.show()

    # same functionality
    average_price_by_bathrooms = housing_three.groupby('bathrooms')['price'].mean()
    plt.figure(figsize=(10, 6))

    # plot line chart
    plt.plot(average_price_by_bathrooms.index, average_price_by_bathrooms.values, marker='o', linestyle='-', color="purple")

    plt.xlabel('# of Bathrooms')
    plt.ylabel('Average Housing Price (in millions)')
    plt.title('Average Housing Prices By Bathrooms')

    if show:
        plt.show()

    return housing_three


# save datasets
# uncomment to unsee plots
housing_one_df = housing_one(show=True)
housing_two_df = housing_two(show=True)
housing_three_df = housing_three(show=True)
# housing_one_df = housing_one()
# housing_two_df = housing_two()
# housing_three_df = housing_three()

print('============ MERGING... ==============')

# merging housing_two_df and housing_three_df based on price
merged = pd.merge(housing_two_df, housing_three_df, on='price', how='outer')

# remove unnecessary columns
merged.drop(columns=['id', 'url', 'region_url'], inplace=True)

# print the merged dataframe based on price
print(merged)

# merging housing_one_df with housing_two_df
merged_ = pd.merge(housing_one_df,
                   housing_two_df,
                   left_on=['median_house_value', 'longitude', 'latitude'],
                   right_on=['price', 'long', 'lat'],
                   how='outer')

# print the merged dataframe
print(merged_)