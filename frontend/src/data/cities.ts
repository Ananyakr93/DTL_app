// Comprehensive list of Indian cities and monitoring stations
// Data sourced from CPCB monitoring network

export interface Station {
    id: string;
    name: string;
    city: string;
    state: string;
    lat: number;
    lon: number;
}

export interface City {
    name: string;
    state: string;
    lat: number;
    lon: number;
    stations: Station[];
}

// Major Indian cities with monitoring stations
export const INDIAN_CITIES: City[] = [
    {
        name: 'Delhi',
        state: 'Delhi',
        lat: 28.6139,
        lon: 77.209,
        stations: [
            { id: 'delhi-anand-vihar', name: 'Anand Vihar', city: 'Delhi', state: 'Delhi', lat: 28.6469, lon: 77.3164 },
            { id: 'delhi-ito', name: 'ITO', city: 'Delhi', state: 'Delhi', lat: 28.6289, lon: 77.2418 },
            { id: 'delhi-punjabi-bagh', name: 'Punjabi Bagh', city: 'Delhi', state: 'Delhi', lat: 28.6683, lon: 77.1319 },
            { id: 'delhi-dwarka', name: 'Dwarka Sector 8', city: 'Delhi', state: 'Delhi', lat: 28.5733, lon: 77.0675 },
            { id: 'delhi-rohini', name: 'Rohini', city: 'Delhi', state: 'Delhi', lat: 28.7328, lon: 77.1167 },
            { id: 'delhi-shadipur', name: 'Shadipur', city: 'Delhi', state: 'Delhi', lat: 28.6514, lon: 77.1474 },
            { id: 'delhi-sirifort', name: 'Siri Fort', city: 'Delhi', state: 'Delhi', lat: 28.5494, lon: 77.2156 },
            { id: 'delhi-r-k-puram', name: 'R K Puram', city: 'Delhi', state: 'Delhi', lat: 28.5633, lon: 77.1833 },
            { id: 'delhi-mandir-marg', name: 'Mandir Marg', city: 'Delhi', state: 'Delhi', lat: 28.6361, lon: 77.2011 },
            { id: 'delhi-nehru-nagar', name: 'Nehru Nagar', city: 'Delhi', state: 'Delhi', lat: 28.5678, lon: 77.2508 },
        ],
    },
    {
        name: 'Mumbai',
        state: 'Maharashtra',
        lat: 19.076,
        lon: 72.8777,
        stations: [
            { id: 'mumbai-bandra', name: 'Bandra', city: 'Mumbai', state: 'Maharashtra', lat: 19.0596, lon: 72.8295 },
            { id: 'mumbai-colaba', name: 'Colaba', city: 'Mumbai', state: 'Maharashtra', lat: 18.9067, lon: 72.8147 },
            { id: 'mumbai-mazgaon', name: 'Mazgaon', city: 'Mumbai', state: 'Maharashtra', lat: 18.9653, lon: 72.8461 },
            { id: 'mumbai-borivali', name: 'Borivali East', city: 'Mumbai', state: 'Maharashtra', lat: 19.2307, lon: 72.8567 },
            { id: 'mumbai-chembur', name: 'Chembur', city: 'Mumbai', state: 'Maharashtra', lat: 19.0522, lon: 72.9005 },
            { id: 'mumbai-worli', name: 'Worli', city: 'Mumbai', state: 'Maharashtra', lat: 19.0176, lon: 72.8153 },
        ],
    },
    {
        name: 'Bangalore',
        state: 'Karnataka',
        lat: 12.9716,
        lon: 77.5946,
        stations: [
            { id: 'bangalore-btm', name: 'BTM Layout', city: 'Bangalore', state: 'Karnataka', lat: 12.9166, lon: 77.6101 },
            { id: 'bangalore-hebbal', name: 'Hebbal', city: 'Bangalore', state: 'Karnataka', lat: 13.0358, lon: 77.5970 },
            { id: 'bangalore-jayanagar', name: 'Jayanagar', city: 'Bangalore', state: 'Karnataka', lat: 12.9308, lon: 77.5838 },
            { id: 'bangalore-peenya', name: 'Peenya', city: 'Bangalore', state: 'Karnataka', lat: 13.0285, lon: 77.5190 },
            { id: 'bangalore-silk-board', name: 'Silk Board', city: 'Bangalore', state: 'Karnataka', lat: 12.9170, lon: 77.6227 },
            { id: 'bangalore-bapuji', name: 'Bapuji Nagar', city: 'Bangalore', state: 'Karnataka', lat: 12.9512, lon: 77.5389 },
        ],
    },
    {
        name: 'Chennai',
        state: 'Tamil Nadu',
        lat: 13.0827,
        lon: 80.2707,
        stations: [
            { id: 'chennai-alandur', name: 'Alandur', city: 'Chennai', state: 'Tamil Nadu', lat: 13.0027, lon: 80.2012 },
            { id: 'chennai-manali', name: 'Manali', city: 'Chennai', state: 'Tamil Nadu', lat: 13.1667, lon: 80.2667 },
            { id: 'chennai-velachery', name: 'Velachery', city: 'Chennai', state: 'Tamil Nadu', lat: 12.9815, lon: 80.2176 },
        ],
    },
    {
        name: 'Kolkata',
        state: 'West Bengal',
        lat: 22.5726,
        lon: 88.3639,
        stations: [
            { id: 'kolkata-jadavpur', name: 'Jadavpur', city: 'Kolkata', state: 'West Bengal', lat: 22.4989, lon: 88.3716 },
            { id: 'kolkata-ballygunge', name: 'Ballygunge', city: 'Kolkata', state: 'West Bengal', lat: 22.5268, lon: 88.3644 },
            { id: 'kolkata-rabindra-bharati', name: 'Rabindra Bharati University', city: 'Kolkata', state: 'West Bengal', lat: 22.6039, lon: 88.3800 },
            { id: 'kolkata-fort-william', name: 'Fort William', city: 'Kolkata', state: 'West Bengal', lat: 22.5558, lon: 88.3428 },
        ],
    },
    {
        name: 'Hyderabad',
        state: 'Telangana',
        lat: 17.385,
        lon: 78.4867,
        stations: [
            { id: 'hyderabad-jubilee-hills', name: 'Jubilee Hills', city: 'Hyderabad', state: 'Telangana', lat: 17.4256, lon: 78.4078 },
            { id: 'hyderabad-sanathnagar', name: 'Sanathnagar', city: 'Hyderabad', state: 'Telangana', lat: 17.4564, lon: 78.4421 },
            { id: 'hyderabad-zoo-park', name: 'Zoo Park', city: 'Hyderabad', state: 'Telangana', lat: 17.3500, lon: 78.4513 },
            { id: 'hyderabad-bolaram', name: 'Bolaram', city: 'Hyderabad', state: 'Telangana', lat: 17.5047, lon: 78.4355 },
        ],
    },
    {
        name: 'Pune',
        state: 'Maharashtra',
        lat: 18.5204,
        lon: 73.8567,
        stations: [
            { id: 'pune-shivajinagar', name: 'Shivajinagar', city: 'Pune', state: 'Maharashtra', lat: 18.5308, lon: 73.8475 },
            { id: 'pune-karve-road', name: 'Karve Road', city: 'Pune', state: 'Maharashtra', lat: 18.5018, lon: 73.8172 },
            { id: 'pune-katraj', name: 'Katraj', city: 'Pune', state: 'Maharashtra', lat: 18.4575, lon: 73.8672 },
        ],
    },
    {
        name: 'Ahmedabad',
        state: 'Gujarat',
        lat: 23.0225,
        lon: 72.5714,
        stations: [
            { id: 'ahmedabad-maninagar', name: 'Maninagar', city: 'Ahmedabad', state: 'Gujarat', lat: 22.9934, lon: 72.6027 },
            { id: 'ahmedabad-chandkheda', name: 'Chandkheda', city: 'Ahmedabad', state: 'Gujarat', lat: 23.1043, lon: 72.5947 },
        ],
    },
    {
        name: 'Jaipur',
        state: 'Rajasthan',
        lat: 26.9124,
        lon: 75.7873,
        stations: [
            { id: 'jaipur-adarsh-nagar', name: 'Adarsh Nagar', city: 'Jaipur', state: 'Rajasthan', lat: 26.9157, lon: 75.7697 },
            { id: 'jaipur-shastri-nagar', name: 'Shastri Nagar', city: 'Jaipur', state: 'Rajasthan', lat: 26.9260, lon: 75.7333 },
        ],
    },
    {
        name: 'Lucknow',
        state: 'Uttar Pradesh',
        lat: 26.8467,
        lon: 80.9462,
        stations: [
            { id: 'lucknow-lalbagh', name: 'Lalbagh', city: 'Lucknow', state: 'Uttar Pradesh', lat: 26.8581, lon: 80.9124 },
            { id: 'lucknow-talkatora', name: 'Talkatora', city: 'Lucknow', state: 'Uttar Pradesh', lat: 26.8544, lon: 80.9269 },
        ],
    },
    // Karnataka cities
    {
        name: 'Mysore',
        state: 'Karnataka',
        lat: 12.2958,
        lon: 76.6394,
        stations: [
            { id: 'mysore-main', name: 'Mysore City', city: 'Mysore', state: 'Karnataka', lat: 12.2958, lon: 76.6394 },
        ],
    },
    {
        name: 'Mangalore',
        state: 'Karnataka',
        lat: 12.9141,
        lon: 74.856,
        stations: [
            { id: 'mangalore-main', name: 'Mangalore City', city: 'Mangalore', state: 'Karnataka', lat: 12.9141, lon: 74.856 },
        ],
    },
    {
        name: 'Hubli',
        state: 'Karnataka',
        lat: 15.3647,
        lon: 75.124,
        stations: [
            { id: 'hubli-main', name: 'Hubli City', city: 'Hubli', state: 'Karnataka', lat: 15.3647, lon: 75.124 },
        ],
    },
    {
        name: 'Chikkamagaluru',
        state: 'Karnataka',
        lat: 13.3161,
        lon: 75.7720,
        stations: [
            { id: 'chikkamagaluru-main', name: 'Chikkamagaluru Town', city: 'Chikkamagaluru', state: 'Karnataka', lat: 13.3161, lon: 75.7720 },
        ],
    },
    {
        name: 'Madikeri',
        state: 'Karnataka',
        lat: 12.4244,
        lon: 75.7382,
        stations: [
            { id: 'madikeri-main', name: 'Madikeri Town', city: 'Madikeri', state: 'Karnataka', lat: 12.4244, lon: 75.7382 },
        ],
    },
    {
        name: 'Shimoga',
        state: 'Karnataka',
        lat: 13.9299,
        lon: 75.5681,
        stations: [
            { id: 'shimoga-main', name: 'Shimoga City', city: 'Shimoga', state: 'Karnataka', lat: 13.9299, lon: 75.5681 },
        ],
    },
    {
        name: 'Belgaum',
        state: 'Karnataka',
        lat: 15.8497,
        lon: 74.4977,
        stations: [
            { id: 'belgaum-main', name: 'Belgaum City', city: 'Belgaum', state: 'Karnataka', lat: 15.8497, lon: 74.4977 },
        ],
    },
    {
        name: 'Davangere',
        state: 'Karnataka',
        lat: 14.4644,
        lon: 75.9218,
        stations: [
            { id: 'davangere-main', name: 'Davangere City', city: 'Davangere', state: 'Karnataka', lat: 14.4644, lon: 75.9218 },
        ],
    },
    // More major cities
    {
        name: 'Kanpur',
        state: 'Uttar Pradesh',
        lat: 26.4499,
        lon: 80.3319,
        stations: [
            { id: 'kanpur-kidwai-nagar', name: 'Kidwai Nagar', city: 'Kanpur', state: 'Uttar Pradesh', lat: 26.4622, lon: 80.3208 },
        ],
    },
    {
        name: 'Nagpur',
        state: 'Maharashtra',
        lat: 21.1458,
        lon: 79.0882,
        stations: [
            { id: 'nagpur-civil-lines', name: 'Civil Lines', city: 'Nagpur', state: 'Maharashtra', lat: 21.1583, lon: 79.0700 },
        ],
    },
    {
        name: 'Indore',
        state: 'Madhya Pradesh',
        lat: 22.7196,
        lon: 75.8577,
        stations: [
            { id: 'indore-vijay-nagar', name: 'Vijay Nagar', city: 'Indore', state: 'Madhya Pradesh', lat: 22.7533, lon: 75.8827 },
        ],
    },
    {
        name: 'Bhopal',
        state: 'Madhya Pradesh',
        lat: 23.2599,
        lon: 77.4126,
        stations: [
            { id: 'bhopal-tt-nagar', name: 'TT Nagar', city: 'Bhopal', state: 'Madhya Pradesh', lat: 23.2333, lon: 77.4000 },
        ],
    },
    {
        name: 'Visakhapatnam',
        state: 'Andhra Pradesh',
        lat: 17.6868,
        lon: 83.2185,
        stations: [
            { id: 'vizag-gvmc', name: 'GVMC Zonal Office', city: 'Visakhapatnam', state: 'Andhra Pradesh', lat: 17.7231, lon: 83.3013 },
        ],
    },
    {
        name: 'Patna',
        state: 'Bihar',
        lat: 25.5941,
        lon: 85.1376,
        stations: [
            { id: 'patna-igsc-planetarium', name: 'IGSC Planetarium', city: 'Patna', state: 'Bihar', lat: 25.6103, lon: 85.1419 },
        ],
    },
    {
        name: 'Vadodara',
        state: 'Gujarat',
        lat: 22.3072,
        lon: 73.1812,
        stations: [
            { id: 'vadodara-main', name: 'Vadodara City', city: 'Vadodara', state: 'Gujarat', lat: 22.3072, lon: 73.1812 },
        ],
    },
    {
        name: 'Ghaziabad',
        state: 'Uttar Pradesh',
        lat: 28.6692,
        lon: 77.4538,
        stations: [
            { id: 'ghaziabad-vasundhara', name: 'Vasundhara', city: 'Ghaziabad', state: 'Uttar Pradesh', lat: 28.6608, lon: 77.3697 },
            { id: 'ghaziabad-indirapuram', name: 'Indirapuram', city: 'Ghaziabad', state: 'Uttar Pradesh', lat: 28.6353, lon: 77.3631 },
        ],
    },
    {
        name: 'Ludhiana',
        state: 'Punjab',
        lat: 30.901,
        lon: 75.8573,
        stations: [
            { id: 'ludhiana-main', name: 'Ludhiana City', city: 'Ludhiana', state: 'Punjab', lat: 30.901, lon: 75.8573 },
        ],
    },
    {
        name: 'Agra',
        state: 'Uttar Pradesh',
        lat: 27.1767,
        lon: 78.0081,
        stations: [
            { id: 'agra-sanjay-place', name: 'Sanjay Place', city: 'Agra', state: 'Uttar Pradesh', lat: 27.1956, lon: 78.0150 },
        ],
    },
    {
        name: 'Nashik',
        state: 'Maharashtra',
        lat: 19.9975,
        lon: 73.7898,
        stations: [
            { id: 'nashik-main', name: 'Nashik City', city: 'Nashik', state: 'Maharashtra', lat: 19.9975, lon: 73.7898 },
        ],
    },
    {
        name: 'Faridabad',
        state: 'Haryana',
        lat: 28.4089,
        lon: 77.3178,
        stations: [
            { id: 'faridabad-sector-16a', name: 'Sector 16A', city: 'Faridabad', state: 'Haryana', lat: 28.4009, lon: 77.3100 },
        ],
    },
    {
        name: 'Meerut',
        state: 'Uttar Pradesh',
        lat: 28.9845,
        lon: 77.7064,
        stations: [
            { id: 'meerut-main', name: 'Meerut City', city: 'Meerut', state: 'Uttar Pradesh', lat: 28.9845, lon: 77.7064 },
        ],
    },
    {
        name: 'Varanasi',
        state: 'Uttar Pradesh',
        lat: 25.3176,
        lon: 82.9739,
        stations: [
            { id: 'varanasi-ardhali-bazar', name: 'Ardhali Bazar', city: 'Varanasi', state: 'Uttar Pradesh', lat: 25.2867, lon: 82.9911 },
        ],
    },
    {
        name: 'Srinagar',
        state: 'Jammu & Kashmir',
        lat: 34.0837,
        lon: 74.7973,
        stations: [
            { id: 'srinagar-main', name: 'Srinagar City', city: 'Srinagar', state: 'Jammu & Kashmir', lat: 34.0837, lon: 74.7973 },
        ],
    },
    {
        name: 'Chandigarh',
        state: 'Chandigarh',
        lat: 30.7333,
        lon: 76.7794,
        stations: [
            { id: 'chandigarh-sector-25', name: 'Sector 25', city: 'Chandigarh', state: 'Chandigarh', lat: 30.7398, lon: 76.7685 },
        ],
    },
    {
        name: 'Guwahati',
        state: 'Assam',
        lat: 26.1445,
        lon: 91.7362,
        stations: [
            { id: 'guwahati-pan-bazar', name: 'Pan Bazar', city: 'Guwahati', state: 'Assam', lat: 26.1839, lon: 91.7455 },
        ],
    },
    {
        name: 'Coimbatore',
        state: 'Tamil Nadu',
        lat: 11.0168,
        lon: 76.9558,
        stations: [
            { id: 'coimbatore-main', name: 'Coimbatore City', city: 'Coimbatore', state: 'Tamil Nadu', lat: 11.0168, lon: 76.9558 },
        ],
    },
    {
        name: 'Thiruvananthapuram',
        state: 'Kerala',
        lat: 8.5241,
        lon: 76.9366,
        stations: [
            { id: 'trivandrum-main', name: 'Thiruvananthapuram City', city: 'Thiruvananthapuram', state: 'Kerala', lat: 8.5241, lon: 76.9366 },
        ],
    },
    {
        name: 'Kochi',
        state: 'Kerala',
        lat: 9.9312,
        lon: 76.2673,
        stations: [
            { id: 'kochi-main', name: 'Kochi City', city: 'Kochi', state: 'Kerala', lat: 9.9312, lon: 76.2673 },
        ],
    },
    {
        name: 'Bhubaneswar',
        state: 'Odisha',
        lat: 20.2961,
        lon: 85.8245,
        stations: [
            { id: 'bhubaneswar-main', name: 'Bhubaneswar City', city: 'Bhubaneswar', state: 'Odisha', lat: 20.2961, lon: 85.8245 },
        ],
    },
    {
        name: 'Ranchi',
        state: 'Jharkhand',
        lat: 23.3441,
        lon: 85.3096,
        stations: [
            { id: 'ranchi-main', name: 'Ranchi City', city: 'Ranchi', state: 'Jharkhand', lat: 23.3441, lon: 85.3096 },
        ],
    },
    {
        name: 'Raipur',
        state: 'Chhattisgarh',
        lat: 21.2514,
        lon: 81.6296,
        stations: [
            { id: 'raipur-main', name: 'Raipur City', city: 'Raipur', state: 'Chhattisgarh', lat: 21.2514, lon: 81.6296 },
        ],
    },
    {
        name: 'Dehradun',
        state: 'Uttarakhand',
        lat: 30.3165,
        lon: 78.0322,
        stations: [
            { id: 'dehradun-main', name: 'Dehradun City', city: 'Dehradun', state: 'Uttarakhand', lat: 30.3165, lon: 78.0322 },
        ],
    },
    {
        name: 'Gurugram',
        state: 'Haryana',
        lat: 28.4595,
        lon: 77.0266,
        stations: [
            { id: 'gurugram-sector-51', name: 'Sector 51', city: 'Gurugram', state: 'Haryana', lat: 28.4332, lon: 77.0535 },
            { id: 'gurugram-teri-gram', name: 'TERI GRAM', city: 'Gurugram', state: 'Haryana', lat: 28.4524, lon: 77.1512 },
        ],
    },
    {
        name: 'Noida',
        state: 'Uttar Pradesh',
        lat: 28.5355,
        lon: 77.391,
        stations: [
            { id: 'noida-sector-62', name: 'Sector 62', city: 'Noida', state: 'Uttar Pradesh', lat: 28.6247, lon: 77.3590 },
            { id: 'noida-sector-125', name: 'Sector 125', city: 'Noida', state: 'Uttar Pradesh', lat: 28.5445, lon: 77.3230 },
        ],
    },
    {
        name: 'Greater Noida',
        state: 'Uttar Pradesh',
        lat: 28.4744,
        lon: 77.504,
        stations: [
            { id: 'greater-noida-main', name: 'Greater Noida', city: 'Greater Noida', state: 'Uttar Pradesh', lat: 28.4744, lon: 77.504 },
        ],
    },
    {
        name: 'Amritsar',
        state: 'Punjab',
        lat: 31.634,
        lon: 74.8723,
        stations: [
            { id: 'amritsar-main', name: 'Amritsar City', city: 'Amritsar', state: 'Punjab', lat: 31.634, lon: 74.8723 },
        ],
    },
    {
        name: 'Jalandhar',
        state: 'Punjab',
        lat: 31.326,
        lon: 75.5762,
        stations: [
            { id: 'jalandhar-main', name: 'Jalandhar City', city: 'Jalandhar', state: 'Punjab', lat: 31.326, lon: 75.5762 },
        ],
    },
    {
        name: 'Jodhpur',
        state: 'Rajasthan',
        lat: 26.2389,
        lon: 73.0243,
        stations: [
            { id: 'jodhpur-main', name: 'Jodhpur City', city: 'Jodhpur', state: 'Rajasthan', lat: 26.2389, lon: 73.0243 },
        ],
    },
    {
        name: 'Udaipur',
        state: 'Rajasthan',
        lat: 24.5854,
        lon: 73.7125,
        stations: [
            { id: 'udaipur-main', name: 'Udaipur City', city: 'Udaipur', state: 'Rajasthan', lat: 24.5854, lon: 73.7125 },
        ],
    },
    {
        name: 'Kota',
        state: 'Rajasthan',
        lat: 25.2138,
        lon: 75.8648,
        stations: [
            { id: 'kota-main', name: 'Kota City', city: 'Kota', state: 'Rajasthan', lat: 25.2138, lon: 75.8648 },
        ],
    },
    {
        name: 'Ajmer',
        state: 'Rajasthan',
        lat: 26.4499,
        lon: 74.6399,
        stations: [
            { id: 'ajmer-main', name: 'Ajmer City', city: 'Ajmer', state: 'Rajasthan', lat: 26.4499, lon: 74.6399 },
        ],
    },
    {
        name: 'Allahabad',
        state: 'Uttar Pradesh',
        lat: 25.4358,
        lon: 81.8463,
        stations: [
            { id: 'allahabad-main', name: 'Prayagraj City', city: 'Allahabad', state: 'Uttar Pradesh', lat: 25.4358, lon: 81.8463 },
        ],
    },
    {
        name: 'Gwalior',
        state: 'Madhya Pradesh',
        lat: 26.2183,
        lon: 78.1828,
        stations: [
            { id: 'gwalior-main', name: 'Gwalior City', city: 'Gwalior', state: 'Madhya Pradesh', lat: 26.2183, lon: 78.1828 },
        ],
    },
    {
        name: 'Jabalpur',
        state: 'Madhya Pradesh',
        lat: 23.1815,
        lon: 79.9864,
        stations: [
            { id: 'jabalpur-main', name: 'Jabalpur City', city: 'Jabalpur', state: 'Madhya Pradesh', lat: 23.1815, lon: 79.9864 },
        ],
    },
    {
        name: 'Vijayawada',
        state: 'Andhra Pradesh',
        lat: 16.5062,
        lon: 80.648,
        stations: [
            { id: 'vijayawada-main', name: 'Vijayawada City', city: 'Vijayawada', state: 'Andhra Pradesh', lat: 16.5062, lon: 80.648 },
        ],
    },
    {
        name: 'Madurai',
        state: 'Tamil Nadu',
        lat: 9.9252,
        lon: 78.1198,
        stations: [
            { id: 'madurai-main', name: 'Madurai City', city: 'Madurai', state: 'Tamil Nadu', lat: 9.9252, lon: 78.1198 },
        ],
    },
    {
        name: 'Tiruchirappalli',
        state: 'Tamil Nadu',
        lat: 10.7905,
        lon: 78.7047,
        stations: [
            { id: 'trichy-main', name: 'Tiruchirappalli City', city: 'Tiruchirappalli', state: 'Tamil Nadu', lat: 10.7905, lon: 78.7047 },
        ],
    },
    {
        name: 'Salem',
        state: 'Tamil Nadu',
        lat: 11.6643,
        lon: 78.146,
        stations: [
            { id: 'salem-main', name: 'Salem City', city: 'Salem', state: 'Tamil Nadu', lat: 11.6643, lon: 78.146 },
        ],
    },
    {
        name: 'Tiruppur',
        state: 'Tamil Nadu',
        lat: 11.1085,
        lon: 77.3411,
        stations: [
            { id: 'tiruppur-main', name: 'Tiruppur City', city: 'Tiruppur', state: 'Tamil Nadu', lat: 11.1085, lon: 77.3411 },
        ],
    },
    {
        name: 'Aurangabad',
        state: 'Maharashtra',
        lat: 19.8762,
        lon: 75.3433,
        stations: [
            { id: 'aurangabad-main', name: 'Aurangabad City', city: 'Aurangabad', state: 'Maharashtra', lat: 19.8762, lon: 75.3433 },
        ],
    },
    {
        name: 'Solapur',
        state: 'Maharashtra',
        lat: 17.6599,
        lon: 75.9064,
        stations: [
            { id: 'solapur-main', name: 'Solapur City', city: 'Solapur', state: 'Maharashtra', lat: 17.6599, lon: 75.9064 },
        ],
    },
    {
        name: 'Thane',
        state: 'Maharashtra',
        lat: 19.2183,
        lon: 72.9781,
        stations: [
            { id: 'thane-main', name: 'Thane City', city: 'Thane', state: 'Maharashtra', lat: 19.2183, lon: 72.9781 },
        ],
    },
    {
        name: 'Navi Mumbai',
        state: 'Maharashtra',
        lat: 19.033,
        lon: 73.0297,
        stations: [
            { id: 'navi-mumbai-main', name: 'Navi Mumbai', city: 'Navi Mumbai', state: 'Maharashtra', lat: 19.033, lon: 73.0297 },
        ],
    },
    {
        name: 'Kalyan-Dombivli',
        state: 'Maharashtra',
        lat: 19.2403,
        lon: 73.1305,
        stations: [
            { id: 'kalyan-main', name: 'Kalyan City', city: 'Kalyan-Dombivli', state: 'Maharashtra', lat: 19.2403, lon: 73.1305 },
        ],
    },
    {
        name: 'Pimpri-Chinchwad',
        state: 'Maharashtra',
        lat: 18.6279,
        lon: 73.8009,
        stations: [
            { id: 'pcmc-main', name: 'PCMC Area', city: 'Pimpri-Chinchwad', state: 'Maharashtra', lat: 18.6279, lon: 73.8009 },
        ],
    },
    // Add more tier 2/3 cities
    {
        name: 'Bhilai',
        state: 'Chhattisgarh',
        lat: 21.2094,
        lon: 81.4285,
        stations: [
            { id: 'bhilai-main', name: 'Bhilai Steel City', city: 'Bhilai', state: 'Chhattisgarh', lat: 21.2094, lon: 81.4285 },
        ],
    },
    {
        name: 'Warangal',
        state: 'Telangana',
        lat: 17.9784,
        lon: 79.5941,
        stations: [
            { id: 'warangal-main', name: 'Warangal City', city: 'Warangal', state: 'Telangana', lat: 17.9784, lon: 79.5941 },
        ],
    },
    {
        name: 'Guntur',
        state: 'Andhra Pradesh',
        lat: 16.3067,
        lon: 80.4365,
        stations: [
            { id: 'guntur-main', name: 'Guntur City', city: 'Guntur', state: 'Andhra Pradesh', lat: 16.3067, lon: 80.4365 },
        ],
    },
    {
        name: 'Nellore',
        state: 'Andhra Pradesh',
        lat: 14.4426,
        lon: 79.9865,
        stations: [
            { id: 'nellore-main', name: 'Nellore City', city: 'Nellore', state: 'Andhra Pradesh', lat: 14.4426, lon: 79.9865 },
        ],
    },
    {
        name: 'Cuttack',
        state: 'Odisha',
        lat: 20.4625,
        lon: 85.8828,
        stations: [
            { id: 'cuttack-main', name: 'Cuttack City', city: 'Cuttack', state: 'Odisha', lat: 20.4625, lon: 85.8828 },
        ],
    },
    {
        name: 'Jamshedpur',
        state: 'Jharkhand',
        lat: 22.8046,
        lon: 86.2029,
        stations: [
            { id: 'jamshedpur-main', name: 'Jamshedpur City', city: 'Jamshedpur', state: 'Jharkhand', lat: 22.8046, lon: 86.2029 },
        ],
    },
    {
        name: 'Dhanbad',
        state: 'Jharkhand',
        lat: 23.7957,
        lon: 86.4304,
        stations: [
            { id: 'dhanbad-main', name: 'Dhanbad City', city: 'Dhanbad', state: 'Jharkhand', lat: 23.7957, lon: 86.4304 },
        ],
    },
    {
        name: 'Asansol',
        state: 'West Bengal',
        lat: 23.6833,
        lon: 86.9667,
        stations: [
            { id: 'asansol-main', name: 'Asansol City', city: 'Asansol', state: 'West Bengal', lat: 23.6833, lon: 86.9667 },
        ],
    },
    {
        name: 'Durgapur',
        state: 'West Bengal',
        lat: 23.5204,
        lon: 87.3119,
        stations: [
            { id: 'durgapur-main', name: 'Durgapur City', city: 'Durgapur', state: 'West Bengal', lat: 23.5204, lon: 87.3119 },
        ],
    },
    {
        name: 'Siliguri',
        state: 'West Bengal',
        lat: 26.7271,
        lon: 88.3953,
        stations: [
            { id: 'siliguri-main', name: 'Siliguri City', city: 'Siliguri', state: 'West Bengal', lat: 26.7271, lon: 88.3953 },
        ],
    },
    {
        name: 'Imphal',
        state: 'Manipur',
        lat: 24.817,
        lon: 93.9368,
        stations: [
            { id: 'imphal-main', name: 'Imphal City', city: 'Imphal', state: 'Manipur', lat: 24.817, lon: 93.9368 },
        ],
    },
    {
        name: 'Shillong',
        state: 'Meghalaya',
        lat: 25.5788,
        lon: 91.8933,
        stations: [
            { id: 'shillong-main', name: 'Shillong City', city: 'Shillong', state: 'Meghalaya', lat: 25.5788, lon: 91.8933 },
        ],
    },
    {
        name: 'Aizawl',
        state: 'Mizoram',
        lat: 23.7271,
        lon: 92.7176,
        stations: [
            { id: 'aizawl-main', name: 'Aizawl City', city: 'Aizawl', state: 'Mizoram', lat: 23.7271, lon: 92.7176 },
        ],
    },
    {
        name: 'Agartala',
        state: 'Tripura',
        lat: 23.8315,
        lon: 91.2868,
        stations: [
            { id: 'agartala-main', name: 'Agartala City', city: 'Agartala', state: 'Tripura', lat: 23.8315, lon: 91.2868 },
        ],
    },
    {
        name: 'Gangtok',
        state: 'Sikkim',
        lat: 27.3389,
        lon: 88.6065,
        stations: [
            { id: 'gangtok-main', name: 'Gangtok City', city: 'Gangtok', state: 'Sikkim', lat: 27.3389, lon: 88.6065 },
        ],
    },
    {
        name: 'Itanagar',
        state: 'Arunachal Pradesh',
        lat: 27.0844,
        lon: 93.6053,
        stations: [
            { id: 'itanagar-main', name: 'Itanagar City', city: 'Itanagar', state: 'Arunachal Pradesh', lat: 27.0844, lon: 93.6053 },
        ],
    },
    {
        name: 'Kohima',
        state: 'Nagaland',
        lat: 25.6751,
        lon: 94.1086,
        stations: [
            { id: 'kohima-main', name: 'Kohima City', city: 'Kohima', state: 'Nagaland', lat: 25.6751, lon: 94.1086 },
        ],
    },
    {
        name: 'Panaji',
        state: 'Goa',
        lat: 15.4909,
        lon: 73.8278,
        stations: [
            { id: 'panaji-main', name: 'Panaji City', city: 'Panaji', state: 'Goa', lat: 15.4909, lon: 73.8278 },
        ],
    },
    {
        name: 'Margao',
        state: 'Goa',
        lat: 15.2832,
        lon: 73.9862,
        stations: [
            { id: 'margao-main', name: 'Margao City', city: 'Margao', state: 'Goa', lat: 15.2832, lon: 73.9862 },
        ],
    },
    {
        name: 'Pondicherry',
        state: 'Puducherry',
        lat: 11.9416,
        lon: 79.8083,
        stations: [
            { id: 'pondicherry-main', name: 'Pondicherry City', city: 'Pondicherry', state: 'Puducherry', lat: 11.9416, lon: 79.8083 },
        ],
    },
    {
        name: 'Silvassa',
        state: 'Dadra and Nagar Haveli',
        lat: 20.2766,
        lon: 73.0063,
        stations: [
            { id: 'silvassa-main', name: 'Silvassa', city: 'Silvassa', state: 'Dadra and Nagar Haveli', lat: 20.2766, lon: 73.0063 },
        ],
    },
    {
        name: 'Daman',
        state: 'Daman and Diu',
        lat: 20.4283,
        lon: 72.8397,
        stations: [
            { id: 'daman-main', name: 'Daman City', city: 'Daman', state: 'Daman and Diu', lat: 20.4283, lon: 72.8397 },
        ],
    },
    {
        name: 'Port Blair',
        state: 'Andaman and Nicobar Islands',
        lat: 11.6234,
        lon: 92.7265,
        stations: [
            { id: 'port-blair-main', name: 'Port Blair', city: 'Port Blair', state: 'Andaman and Nicobar Islands', lat: 11.6234, lon: 92.7265 },
        ],
    },
];

// Get all stations as a flat array
export function getAllStations(): Station[] {
    return INDIAN_CITIES.flatMap((city) => city.stations);
}

// Get all city names
export function getAllCityNames(): string[] {
    return INDIAN_CITIES.map((city) => city.name);
}

// Search cities and stations
export function searchLocations(query: string): Array<{ type: 'city' | 'station'; name: string; city?: string; state: string; id?: string }> {
    const q = query.toLowerCase().trim();
    if (q.length < 2) return [];

    const results: Array<{ type: 'city' | 'station'; name: string; city?: string; state: string; id?: string }> = [];

    for (const city of INDIAN_CITIES) {
        // Check city name
        if (city.name.toLowerCase().includes(q)) {
            results.push({ type: 'city', name: city.name, state: city.state });
        }

        // Check stations
        for (const station of city.stations) {
            if (station.name.toLowerCase().includes(q)) {
                results.push({ type: 'station', name: station.name, city: city.name, state: city.state, id: station.id });
            }
        }
    }

    return results.slice(0, 15); // Limit results
}

// Get city by name
export function getCityByName(name: string): City | undefined {
    return INDIAN_CITIES.find((city) => city.name.toLowerCase() === name.toLowerCase());
}

// Get station by ID
export function getStationById(id: string): Station | undefined {
    return getAllStations().find((station) => station.id === id);
}

// Find nearest city to coordinates
export function findNearestCity(lat: number, lon: number): City {
    let nearest = INDIAN_CITIES[0];
    let minDist = Infinity;

    for (const city of INDIAN_CITIES) {
        const dist = Math.sqrt(Math.pow(city.lat - lat, 2) + Math.pow(city.lon - lon, 2));
        if (dist < minDist) {
            minDist = dist;
            nearest = city;
        }
    }

    return nearest;
}
