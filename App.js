// // App.js
// import React, { useState, useEffect } from 'react';
// import { View, Text,Image,Button,  StyleSheet ,Alert, ScrollView,TouchableOpacity} from 'react-native';
// import { Linking } from 'react-native';
// // import * as SMS from 'expo-sms';
// import { sendSMS } from 'react-native-sms';
// // import SplashScreen from 'react-native-splash-screen';
// import  data from './data.json';
// // import { NavigationContainer } from '@react-navigation/native';
// // import { createStackNavigator } from '@react-navigation/stack';


// const App = () => {
//   const [label,setLabel] = useState(0);
//   const [heartbeat, setHeartbeat] = useState(0);
//   const [eda, setEda] = useState(0);
//   const [temperature, setTemperature] = useState(0);
//   const [timestamp, setTimestamp] = useState('');

//   useEffect(() => {
//     // Read data from file
//     console.log('Label from data:', data.label); // Check the value of label
//     setLabel(data.label);
//     setHeartbeat(data.heartbeat);
//     setTemperature(data.temperature)
//     setTimestamp(data.timestamp);
//     setEda(data.eda);
// }, []);
// // Function to handle help button press
// const handleHelpButtonPress = () => {
//   Alert.alert(
//     'Tips for Managing Stress During Pregnancy',
//     '1. Practice deep breathing exercises, taking slow, deep breaths to calm the nervous system and promote relaxation.' + '\n' +
//     '2. Engage in gentle physical activity, such as prenatal yoga, walking, or swimming, to release endorphins and reduce tension.' + '\n' +
//     '3. Take short breaks throughout the day to rest and recharge and prioritizing self care and relaxation.' + '\n' +
//     '4. Use mindfulness techniques to stay present and grounded, focusing on the sensations of the moment rather than worries or concerns.' + '\n' +
//     '5. Listen to soothing music or nature sounds to create a calming environment and reduce stress levels.' + '\n' +
//     '6. Connect with supportive friends, family members, or a prenatal support group to share experiences and receive encouragement.' + '\n' +
//     '7. Practice positive affirmations, repeating phrases that affirm strength, resilience, and confidence in the bodys ability to nurture and birth a baby.' + '\n' +
//     '8. Spend time outdoors in nature, enjoying fresh air, sunlight, and the calming effects of natural surroundings.' + '\n' +
//     '9. Engage in creative activities that bring joy and relaxation, such as drawing, journaling, or crafting.' + '\n' +
//     '10. Seek professional support if feeling overwhelmed or experiencing persistent stress or anxiety during pregnancy, reaching out to a healthcare provider or mental health professional for guidance and assistance.'
//   );
// };
// // Define the numbers to send SOS messages to

// const handleEmergencyButtonPress = async () => {
// const emergencyNumbers = ['0987654321', '0987654321']; // Example emergency numbers

//   // Call emergency number
//   Linking.openURL('tel:9111'); // Replace '911' with the actual emergency number
  
//     // Send SOS messages
//     const message = 'Emergency: Please help!';
//     try {
//       for (const number of emergencyNumbers) {
//         await sendSMS(number, message); // Sends SMS to each number in emergencyNumbers array
//       }
//       console.log('SOS messages sent successfully');
//     } catch (error) {
//       console.error('Error sending SOS messages:', error);
//       // You can handle the error here, such as displaying an alert to the user
//     }
  
// };

// // Function to render different UI elements based on label value
// const renderUIBasedOnLabel = () => {
//   switch (label) {
//     case 0:
//       return (
//         <Text style={styles.infoText}>Everything is fine</Text>
//       );
//     case 1:
//       return (
//         <View>
//           <Text style={styles.infoText}>Request Help</Text>
//           <Button title="Help" onPress={handleHelpButtonPress} />
//         </View>
//       );
//     case 2:
//       // handleEmergencyButtonPress()
//       setTimeout(handleEmergencyButtonPress, Math.floor(Math.random() * 3000) + 3000); // Delay between 3 to 5 seconds
//       return (
//         <View>
//           <Text style={styles.infoText}>Emergency</Text>
//           <Button title="Emergency" onPress={handleEmergencyButtonPress} />
//         </View>
//       );
//     default:
//       return null;
//   }
// };

//   return (
//     // <ScrollView>
//       <View style={styles.container}>
//       {/* Data Display */}
//       <Text style={styles.header}>NURTUREHUB</Text>
//       <View style={styles.dataContainer}>
//         <Text style={styles.subheading}>Data</Text>
//         <View style={styles.gridContainer}>
//           <View style={styles.gridItem}>
//             <Image source={require('./images/heart.jpg')} style={styles.icon} />
//             <Text>Heartbeat: {heartbeat}</Text>
//           </View>
//           <View style={styles.gridItem}>
//             <Image source={require('./images/1995.jpg')} style={styles.icon} />
//             <Text>EDA: {eda}</Text>
//           </View>
//           <View style={styles.gridItem}>
//             <Image source={require('./images/thermometers_hot_and_cold.jpg')} style={styles.icon} />
//             <Text>X: {temperature}</Text>
//           </View>
//           <View style={styles.gridItem}>
//             <Image source={require('./images/clock-front-side-with-white-background.jpg')} style={styles.icon} />
//             <Text>Timestamp: {timestamp}</Text>
//           </View>
//         </View>
//       </View>
//        <View style={styles.stressIndicator}>
//         <Text>Stress Level: {label}</Text>
//        </View>
//        {/* Display UI based on label */}
//       <View style={styles.labelContainer}>
//         {renderUIBasedOnLabel(label)}
//       </View>
//     </View>
//     // </ScrollView>
//   );
// };


// const styles = StyleSheet.create({
//   header: {
//     justifyContent: 'center',
//     paddingBottom: 100,
//     fontSize: 40,
//     color: 'white'
//   },
//   container: {
//     flex: 1,
//     justifyContent: 'center',
//     alignItems: 'center',
//     backgroundColor: '#ffb6c1',
//   },
//   dataContainer: {
//     marginBottom: 20,
//     width: '80%',
//   },
//   subheading: {
//     fontSize: 20,
//     fontWeight: 'bold',
//     marginBottom: 20,
//     color: '#333',
//   },
//   gridContainer: {
//     flexDirection: 'row',
//     flexWrap: 'wrap',
//     justifyContent: 'space-between',
//     marginLeft:50
//   },
//   gridItem: {
//     width: '48%',
//     marginBottom: 10,
//     paddingBottom: 100,
//   },
//   icon: {
//     width: 50,
//     height: 50,
//     marginBottom: 5,
//   },
//   stressIndicator: {
//     alignItems: 'center',
//   },
//   infoText: {
//     fontSize: 18,
//     fontWeight: 'bold',
//     marginBottom: 10,
//   },
//   labelContainer: {
//     marginTop: 20,
//     alignItems: 'center',
//   },
// });

// export default App;












// import React from "react";
// import { View, Text, StyleSheet, ScrollView, Image, Button } from 'react-native';
// import Heart from './Images11/heart-rate.jpg'
// import Water from './Images11/recycling-water.jpg'
// import Temperature from './Images11/high-temperature.jpg'
// import Timer from './Images11/3d-stopwatch.jpg'
// import Help from './Images11/help.jpg'
// // import Stress from '../Assets/Images/stress.png'
// import Depression from './Images11/depression.jpg'

// const InternUi = () => {
//     return (
//         <ScrollView style={styles.Container}>
//             <View style={styles.titleContainer}>
//                 <Text style={styles.titleText}>NURTUREHUB</Text>
//             </View>
//             <View style={styles.headerContainer}>
//                 <Text style={styles.headerText}>Good Afternoon, Rihan</Text>
//             </View>
//             <View style={styles.bottomContainer}>
//                 <View style={styles.bottomChildContainer}>
//                     <View style={styles.iconContainer}>
//                         <Image source={Heart} style={styles.iconStyle} />
//                     </View>
//                     <View style={styles.textContainer}>
//                         <Text style={styles.rightTextStyle}>Heartbeat: 99.43</Text>
//                     </View>
//                 </View>
//                 <View style={styles.bottomChildContainer}>
//                     <View style={styles.iconContainer}>
//                         <Image source={Water} style={styles.iconStyle} />
//                     </View>
//                     <View style={styles.textContainer}>
//                         <Text style={styles.rightTextStyle}>EDA: 99.43</Text>
//                     </View>
//                 </View>
//                 <View style={styles.bottomChildContainer}>
//                     <View style={styles.iconContainer}>
//                         <Image source={Temperature} style={styles.iconStyle} />
//                     </View>
//                     <View style={styles.textContainer}>
//                         <Text style={styles.rightTextStyle}>X: 99.43</Text>
//                     </View>
//                 </View>
//                 <View style={styles.bottomChildContainer}>
//                     <View style={styles.iconContainer}>
//                         <Image source={Timer} style={styles.iconStyle} />
//                     </View>
//                     <View style={styles.textContainer}>
//                         <Text style={styles.rightTextStyle}>TimeStamp: 99.43</Text>
//                     </View>
//                 </View>
//                 <View style={styles.footerContainer}>
//                     <Image source={Depression} style={styles.helpIconStyle} />
//                     <Text style={styles.footerText}>Stress Level: 1</Text>
//                 </View>
//                 <View style={styles.helpContainer}>
//                     <Image source={Help} style={styles.helpIconStyle} />
//                     <View style={styles.helpIconContainer}>
//                         {/* {renderUIBasedOnLabel(label)} */}

//                         {/* UNCOMMENT THE ABOVE CODE AND REMOVE THIS BELOW BUTTON */}
//                         <Button title="help" />
//                     </View>
//                 </View>
//             </View>
//         </ScrollView>
//     )
// }

// const styles = StyleSheet.create({
//     Container: {
//         flex: 1,
//         backgroundColor: '#F3F8FF'
//     },
//     titleContainer: {
//         flex: 1,
//         backgroundColor: '#ffffff',
//         justifyContent: 'center',
//         alignItems: 'center',
//         paddingTop: 15,
//         paddingBottom: 15,
//         elevation: 5,
//         borderBottomWidth: 1,
//         borderBottomColor: '#BED7DC'
//     },
//     titleText: {
//         fontSize: 25,
//         fontWeight: 'bold',
//         color: '#1E0342'
//     },
//     headerContainer: {
//         backgroundColor: '#ffffff',
//         margin: 15,
//         borderRadius: 5,
//         height: 70,
//         flex: 1,
//         alignItems: 'flex-start',
//         justifyContent: 'center',
//         paddingLeft: 15,
//         elevation: 5
//     },
//     headerText: {
//         fontSize: 17,
//         fontWeight: 'bold',
//         color: '#3A4750'
//     },
//     bottomContainer: {
//         flex: 1,
//         backgroundColor: '#ffffff',
//         marginLeft: 15,
//         marginRight: 15,
//         borderRadius: 5,
//         elevation: 5,
//         padding: 15,
//         marginBottom: 15
//     },
//     iconContainer: {
//         backgroundColor: '#F5F4F4',
//         width: 90,
//         height: 90,
//         borderRadius: 50,
//         alignItems: 'center',
//         justifyContent: 'center',
//         borderWidth: 5,
//         borderColor: '#E1EEFF'
//     },
//     iconStyle: {
//         width: 30,
//         height: 30
//     },
//     helpIconStyle: {
//         width: 60,
//         height: 60
//     },
//     bottomChildContainer: {
//         flex: 1,
//         flexDirection: 'row',
//         alignItems: 'center',
//         marginBottom: 30,
//         elevation: 5
//     },
//     textContainer: {
//         flex: 1,
//         backgroundColor: '#E1EEFF',
//         borderRadius: 5,
//         marginLeft: -45,
//         zIndex: -1,
//         justifyContent: 'center',
//         alignItems: 'center',
//         height: 115,
//     },
//     rightTextStyle: {
//         color: '#10316B',
//         fontSize: 20,
//         fontWeight: 'bold',
//         width: 190
//     },
//     footerContainer: {
//         flex: 1,
//         backgroundColor: '#ffffff',
//         elevation: 5,
//         paddingTop: 20,
//         paddingBottom: 20,
//         justifyContent: 'center',
//         alignItems: 'center',
//         borderWidth: 2,
//         borderColor: '#B3C8CF',
//         borderRadius: 10
//     },
//     footerText: {
//         color: '#102C57',
//         fontSize: 15,
//         fontWeight: 'bold',
//         marginTop: 10
//     },
//     helpContainer: {
//         flex: 1,
//         backgroundColor: 'red',
//         alignItems: 'center',
//         justifyContent: 'center',
//         backgroundColor: '#ffffff',
//         elevation: 5,
//         paddingTop: 20,
//         paddingBottom: 20,
//         marginTop: 20,
//         borderWidth: 2,
//         borderColor: '#B3C8CF',
//         borderRadius: 10
//     },
//     helpIconContainer: {
//         marginTop: 10
//     }
// })
// export default InternUi;






import React, { useState, useEffect } from 'react';
import { View, Text, Image, Button, StyleSheet, Alert, ScrollView } from 'react-native';
import Heart from './Images11/heart-rate.jpg';
import Water from './Images11/recycling-water.jpg';
import Temperature from './Images11/high-temperature.jpg';
import Timer from './Images11/3d-stopwatch.jpg';
import Help from './Images11/help.jpg';
import Depression from './Images11/depression.jpg';
import data from './data.json';
import { Linking } from 'react-native';

const InternUi = () => {
    const [label, setLabel] = useState(0);
    const [heartbeat, setHeartbeat] = useState(0);
    const [eda, setEda] = useState(0);
    const [temperature, setTemperature] = useState(0);
    const [timestamp, setTimestamp] = useState('');

    useEffect(() => {
        setLabel(data.label);
        setHeartbeat(data.heartbeat);
        setTemperature(data.temperature);
        setTimestamp(data.timestamp);
        setEda(data.eda);
    }, []);

    // Function to handle help button press 
    const handleHelpButtonPress = () => {
        Alert.alert(
            'Tips for Managing Stress During Pregnancy',
            '1. Practice deep breathing exercises, taking slow, deep breaths to calm the nervous system and promote relaxation.\n\n' +
            '2. Engage in gentle physical activity, such as prenatal yoga, walking, or swimming, to release endorphins and reduce tension.\n\n' +
            '3. Take short breaks throughout the day to rest and recharge and prioritizing self-care and relaxation.\n\n' +
            '4. Use mindfulness techniques to stay present and grounded, focusing on the sensations of the moment rather than worries or concerns.\n\n' +
            '5. Listen to soothing music or nature sounds to create a calming environment and reduce stress levels.\n\n' +
            '6. Connect with supportive friends, family members, or a prenatal support group to share experiences and receive encouragement.\n\n' +
            '7. Practice positive affirmations, repeating phrases that affirm strength, resilience, and confidence in the body\'s ability to nurture and birth a baby.\n\n' +
            '8. Spend time outdoors in nature, enjoying fresh air, sunlight, and the calming effects of natural surroundings.\n\n' +
            '9. Engage in creative activities that bring joy and relaxation, such as drawing, journaling, or crafting.\n\n' +
            '10. Seek professional support if feeling overwhelmed or experiencing persistent stress or anxiety during pregnancy, reaching out to a healthcare provider or mental health professional for guidance and assistance.'
        );
    };
    

    // Function to handle emergency button press
    const handleEmergencyButtonPress = async () => {
        // const emergencyNumbers = ['0987654321', '0987654321']; // Example emergency numbers
        const emergencyNumbers='9591254722';
        // Call emergency number
        Linking.openURL('tel:9535969254'); // Replace '911' with the actual emergency number

        // Send SOS messages
        const message = 'Emergency: Please help!';
        try {
            for (const number of emergencyNumbers) {
                await sendSMS(number, message); // Sends SMS to each number in emergencyNumbers array
            }
            console.log('SOS messages sent successfully');
        } catch (error) {
            console.error('Error sending SOS messages:', error);
            // You can handle the error here, such as displaying an alert to the user
        }
    };

    // Function to render different UI elements based on label value
    const renderUIBasedOnLabel = () => {
        switch (label) {
            case 0:
                return (
                    <Text style={styles.infoText}>Everything is fine</Text>
                );
            case 1:
                return (
                    <View>
                        <Text style={styles.infoText}>Request Help</Text>
                        <Button title="Help" onPress={handleHelpButtonPress} />
                    </View>
                );
            case 2:
                setTimeout(handleEmergencyButtonPress, Math.floor(Math.random() * 3000) + 3000); // Delay between 3 to 5 seconds
                return (
                    <View>
                        {/* <Text style={styles.infoText}>Emergency</Text> */}
                        <Button title="Emergency" onPress={handleEmergencyButtonPress} />
                    </View>
                );
            default:
                return null;
        }
    };

    return (
        <ScrollView style={styles.Container}>
            <View style={styles.titleContainer}>
                <Text style={styles.titleText}>NURTURHUB</Text>
            </View>
            <View style={styles.headerContainer}>
                <Text style={styles.headerText}>Good Afternoon, Rihan</Text>
            </View>
            <View style={styles.bottomContainer}>
                <View style={styles.bottomChildContainer}>
                    <View style={styles.iconContainer}>
                        <Image source={Heart} style={styles.iconStyle} />
                    </View>
                    <View style={styles.textContainer}>
                        <Text style={styles.rightTextStyle}>Heartbeat: {heartbeat}</Text>
                    </View>
                </View>
                <View style={styles.bottomChildContainer}>
                    <View style={styles.iconContainer}>
                        <Image source={Water} style={styles.iconStyle} />
                    </View>
                    <View style={styles.textContainer}>
                        <Text style={styles.rightTextStyle}>EDA: {eda}</Text>
                    </View>
                </View>
                <View style={styles.bottomChildContainer}>
                    <View style={styles.iconContainer}>
                        <Image source={Temperature} style={styles.iconStyle} />
                    </View>
                    <View style={styles.textContainer}>
                        <Text style={styles.rightTextStyle}>Temperature: {temperature}</Text>
                    </View>
                </View>
                <View style={styles.bottomChildContainer}>
                    <View style={styles.iconContainer}>
                        <Image source={Timer} style={styles.iconStyle} />
                    </View>
                    <View style={styles.textContainer}>
                        <Text style={styles.rightTextStyle}>TimeStamp: {timestamp}</Text>
                    </View>
                </View>
                <View style={styles.footerContainer}>
                    <Image source={Depression} style={styles.helpIconStyle} />
                    <Text style={styles.footerText}>Stress Level: {label}</Text>
                </View>
                <View style={styles.helpContainer}>
                    <Image source={Help} style={styles.helpIconStyle} />
                    <View style={styles.helpIconContainer}>
                        {renderUIBasedOnLabel()}
                    </View>
                </View>
            </View>
        </ScrollView>
    )
}

const styles = StyleSheet.create({
    Container: {
                flex: 1,
                backgroundColor: '#F3F8FF'
            },
            titleContainer: {
                flex: 1,
                backgroundColor: '#ffffff',
                justifyContent: 'center',
                alignItems: 'center',
                paddingTop: 15,
                paddingBottom: 15,
                elevation: 5,
                borderBottomWidth: 1,
                borderBottomColor: '#BED7DC'
            },
            titleText: {
                fontSize: 25,
                fontWeight: 'bold',
                color: '#1E0342'
            },
            headerContainer: {
                backgroundColor: '#ffffff',
                margin: 15,
                borderRadius: 5,
                height: 70,
                flex: 1,
                alignItems: 'flex-start',
                justifyContent: 'center',
                paddingLeft: 15,
                elevation: 5
            },
            headerText: {
                fontSize: 17,
                fontWeight: 'bold',
                color: '#3A4750'
            },
            bottomContainer: {
                flex: 1,
                backgroundColor: '#ffffff',
                marginLeft: 15,
                marginRight: 15,
                borderRadius: 5,
                elevation: 5,
                padding: 15,
                marginBottom: 15
            },
            iconContainer: {
                backgroundColor: '#F5F4F4',
                width: 90,
                height: 90,
                borderRadius: 50,
                alignItems: 'center',
                justifyContent: 'center',
                borderWidth: 5,
                borderColor: '#E1EEFF'
            },
            iconStyle: {
                width: 30,
                height: 30
            },
            helpIconStyle: {
                width: 60,
                height: 60
            },
            bottomChildContainer: {
                flex: 1,
                flexDirection: 'row',
                alignItems: 'center',
                marginBottom: 30,
                elevation: 5
            },
            textContainer: {
                flex: 1,
                backgroundColor: '#E1EEFF',
                borderRadius: 5,
                marginLeft: -45,
                zIndex: -1,
                justifyContent: 'center',
                alignItems: 'center',
                height: 115,
                paddingLeft:50
            },
            rightTextStyle: {
                color: '#10316B',
                fontSize: 20,
                fontWeight: 'bold',
                width: 190
            },
            footerContainer: {
                flex: 1,
                backgroundColor: '#ffffff',
                elevation: 5,
                paddingTop: 20,
                paddingBottom: 20,
                justifyContent: 'center',
                alignItems: 'center',
                borderWidth: 2,
                borderColor: '#B3C8CF',
                borderRadius: 10
            },
            footerText: {
                color: '#102C57',
                fontSize: 15,
                fontWeight: 'bold',
                marginTop: 10
            },
            helpContainer: {
                flex: 1,
                backgroundColor: 'red',
                alignItems: 'center',
                justifyContent: 'center',
                backgroundColor: '#ffffff',
                elevation: 5,
                paddingTop: 20,
                paddingBottom: 20,
                marginTop: 20,
                borderWidth: 2,
                borderColor: '#B3C8CF',
                borderRadius: 10
            },
            helpIconContainer: {
                marginTop: 10
            }
        });

export default InternUi;

